// Copyright 2026 The go-llama.cpp Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package llama

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"iter"
	"mime"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"

	sse "github.com/tmaxmax/go-sse"
)

// ErrStreamConsumed is yielded when a stream returned by Stream is ranged over a second time. A stream sends its
// request when the range loop starts and can be consumed only once.
var ErrStreamConsumed = errors.New("llama: stream already consumed")

// ErrIncompleteStream means a stream ended without its endpoint's terminal frame or frames.
var ErrIncompleteStream = errors.New("llama: incomplete event stream")

// streamDone is the data of the event that ends the streams of the OpenAI-compatible completion endpoints.
const streamDone = "[DONE]"

// Stream sends a request whose response is a server-sent event stream and returns its events decoded into T.
//
// Nothing is sent until the returned sequence is ranged over. The response body is closed when the stream ends
// (a "data: [DONE]" event or the end of the body), when an error is yielded, when the loop breaks, and when ctx is
// canceled. Ranging over the sequence a second time sends nothing and yields ErrStreamConsumed once. The request
// is retried under the client's policy only until a response is accepted; after that no retry happens.
//
// Events carrying an error, either "data: {"error": {...}}" or an event named "error", are yielded as an
// *APIError and end the stream; their StatusCode is the status of the stream response. Comment lines (keep-alive
// pings) are skipped. A body that ends inside an event yields an error wrapping io.ErrUnexpectedEOF.
// A completion, infill, Responses or Messages stream that ends before its terminal frame yields
// ErrIncompleteStream. The /models/sse subscription has no terminal frame, so EOF on server shutdown
// ends it normally. Unknown endpoints also accept EOF at a frame boundary.
func Stream[T any](ctx context.Context, c *Client, method, path string, body any, opts ...RequestOption) iter.Seq2[*T, error] {
	return singleUseStream(func(yield func(*T, error) bool) {
		resp, err := c.do(ctx, method, path, body, "text/event-stream", opts)
		if err != nil {
			yield(nil, err)
			return
		}
		defer resp.Body.Close()
		if err := checkEventStream(resp); err != nil {
			yield(nil, c.RedactError(err))
			return
		}
		end := endpointStreamEnd(path)
		if end.marker == nativeFinal {
			end.setNativeRequest(nativeStreamRequest(body, resp.Request))
		}
		for v, err := range readStream[T](resp.Body, c.maxEventSize, resp.StatusCode, resp.Header, end) {
			if err != nil && ctx.Err() != nil {
				err = fmt.Errorf("llama: stream interrupted: %w", context.Cause(ctx))
			}
			err = c.RedactError(err)
			if !yield(v, err) || err != nil {
				return
			}
		}
	})
}

func singleUseStream[T any](seq iter.Seq2[*T, error]) iter.Seq2[*T, error] {
	var used atomic.Bool
	return func(yield func(*T, error) bool) {
		if used.Swap(true) {
			yield(nil, ErrStreamConsumed)
			return
		}
		seq(yield)
	}
}

// DecodeStream decodes SSE framing without checking for an endpoint-specific terminal frame. Use
// DecodeEndpointStream to distinguish a complete response from a recording cut at a frame boundary.
// maxEventSize limits the size of one event; zero or less selects DefaultMaxEventSize. Errors carried
// by events are yielded as *APIError with a zero StatusCode.
func DecodeStream[T any](r io.Reader, maxEventSize int) iter.Seq2[*T, error] {
	return readStream[T](r, maxEventSize, 0, nil, streamEnd{})
}

// DecodeEndpointStream decodes recorded events with the same terminal checks as Stream for path.
// For /completion and /infill, expectedResults is the number of prompts times the number of completions
// per prompt; zero checks only for results observable in the stream. Pass the request's response_fields
// as responseFields when replaying a projected native stream. Other endpoints ignore these arguments.
// The reader and event size limit follow DecodeStream.
func DecodeEndpointStream[T any](r io.Reader, maxEventSize int, path string, expectedResults int, responseFields ...string) iter.Seq2[*T, error] {
	end := endpointStreamEnd(path)
	end.setNativeRequest(expectedResults, responseFields)
	return readStream[T](r, maxEventSize, 0, nil, end)
}

const nativeFinal = "native completion result"

type streamEnd struct {
	marker              string
	expected            int
	indexes             map[int]bool
	anonymous           int
	omitStop, omitIndex bool
	lastFinal           bool
}

func (e *streamEnd) setNativeRequest(expected int, fields []string) {
	e.expected = expected
	e.omitStop = len(fields) > 0 && !slices.Contains(fields, "stop")
	e.omitIndex = len(fields) > 0 && !slices.Contains(fields, "index")
}

func endpointStreamEnd(path string) streamEnd {
	path, _, _ = strings.Cut(path, "?")
	switch path {
	case "/completion", "/completions", "/infill":
		return streamEnd{marker: nativeFinal}
	case "/v1/completions", "/chat/completions", "/v1/chat/completions":
		return streamEnd{marker: streamDone}
	case "/responses", "/v1/responses":
		return streamEnd{marker: "response.completed"}
	case "/v1/messages":
		return streamEnd{marker: "message_stop"}
	default:
		return streamEnd{}
	}
}

func (e *streamEnd) observe(event string, value any, data []byte) error {
	if e.marker != nativeFinal {
		e.lastFinal = event == e.marker
		return nil
	}
	var stop *bool
	var index *int
	if result, ok := value.(*CompletionResponse); ok {
		stop, index = result.Stop, result.Index
	} else {
		var frame struct {
			Stop  *bool `json:"stop"`
			Index *int  `json:"index"`
		}
		if err := json.Unmarshal(data, &frame); err != nil {
			return err
		}
		stop, index = frame.Stop, frame.Index
	}
	// Only final frames undergo response_fields projection; every partial frame carries stop:false.
	e.lastFinal = ((stop == nil && e.omitStop) || (stop != nil && *stop)) && (index != nil || e.omitIndex)
	if index != nil {
		if e.indexes == nil {
			e.indexes = make(map[int]bool)
		}
		e.indexes[*index] = e.lastFinal
	} else if e.lastFinal {
		e.anonymous++
	}
	return nil
}

func (e *streamEnd) missing() error {
	switch e.marker {
	case "":
		return nil
	case nativeFinal:
		final := e.anonymous
		for _, stopped := range e.indexes {
			if stopped {
				final++
			}
		}
		want := max(1, e.expected, len(e.indexes))
		if !e.lastFinal || final < want {
			return fmt.Errorf("%w: received %d final native results, expected at least %d and a final frame last", ErrIncompleteStream, final, want)
		}
	default:
		if !e.lastFinal || e.marker == streamDone {
			return fmt.Errorf("%w: missing %s", ErrIncompleteStream, e.marker)
		}
	}
	return nil
}

// checkEventStream rejects a successful response that is not an event stream, for example a JSON body returned
// because the request did not ask for streaming.
func checkEventStream(resp *http.Response) error {
	ct := resp.Header.Get("Content-Type")
	mt, _, err := mime.ParseMediaType(ct)
	if err == nil && mt == "text/event-stream" {
		return nil
	}
	return fmt.Errorf("llama: expected a text/event-stream response, got Content-Type %q", ct)
}

// readStream decodes the events of r. status and header are copied into the APIError of an error event.
func readStream[T any](r io.Reader, maxEventSize, status int, header http.Header, rule streamEnd) iter.Seq2[*T, error] {
	if maxEventSize <= 0 {
		maxEventSize = DefaultMaxEventSize
	}
	return func(yield func(*T, error) bool) {
		end := rule
		tr := &tailReader{r: r}
		for ev, err := range sse.Read(tr, &sse.ReadConfig{MaxEventSize: maxEventSize}) {
			if err != nil {
				if missing := end.missing(); errors.Is(err, io.ErrUnexpectedEOF) && missing != nil {
					yield(nil, fmt.Errorf("%w: %w", missing, err))
				} else {
					yield(nil, fmt.Errorf("llama: reading event stream: %w", err))
				}
				return
			}
			if ev.Type == "error" {
				yield(nil, eventError(ev.Data, status, header))
				return
			}
			if ev.Data == "" {
				continue
			}
			if ev.Data == streamDone {
				if end.marker != "" && end.marker != streamDone {
					yield(nil, fmt.Errorf("%w: unexpected [DONE], want %s", ErrIncompleteStream, end.marker))
				}
				return
			}
			data := []byte(ev.Data)
			if apiErr := inlineError(data, status, header); apiErr != nil {
				yield(nil, apiErr)
				return
			}
			v := new(T)
			if err := json.Unmarshal(data, v, jsonOptions); err != nil {
				yield(nil, fmt.Errorf("llama: decoding event: %w", err))
				return
			}
			if ev.Data == "null" {
				// A native stream announces each result after the first with a bare null
				// (server_task_result_cmpl_partial::to_json, server-task.cpp:1031-1033): a result
				// began and its final frame is still owed.
				end.lastFinal = false
			} else if err := end.observe(ev.Type, v, data); err != nil {
				yield(nil, fmt.Errorf("llama: decoding native stream fields: %w", err))
				return
			}
			if !yield(v, nil) {
				return
			}
		}
		if tr.truncated() {
			yield(nil, fmt.Errorf("%w: stream ended inside an event: %w", ErrIncompleteStream, io.ErrUnexpectedEOF))
		} else if err := end.missing(); err != nil {
			yield(nil, err)
		}
	}
}

// eventError converts the data of an event named "error", which carries the bare error object.
func eventError(data string, status int, header http.Header) *APIError {
	raw := []byte(data)
	var obj errorObject
	if err := json.Unmarshal(raw, &obj, jsonOptions); err != nil {
		return &APIError{StatusCode: status, Message: bodySnippet(raw, status), Raw: raw, Header: header}
	}
	// Some producers wrap the object in {"error": ...} even on a named error event.
	if obj.Type == "" && obj.Message == "" {
		if apiErr := inlineError(raw, status, header); apiErr != nil {
			return apiErr
		}
	}
	return newAPIError(status, &obj, raw, header)
}

// errorProbe reads only the "error" member of an event.
type errorProbe struct {
	Error jsontext.Value `json:"error"`
}

// inlineError returns the error carried by a data-only event shaped {"error": {...}}, or nil for any other event.
func inlineError(data []byte, status int, header http.Header) *APIError {
	if !bytes.Contains(data, []byte(`"error"`)) {
		return nil
	}
	var probe errorProbe
	if err := json.Unmarshal(data, &probe); err != nil || probe.Error.Kind() != '{' {
		return nil
	}
	var obj errorObject
	if err := json.Unmarshal(probe.Error, &obj, jsonOptions); err != nil {
		return nil
	}
	return newAPIError(status, &obj, data, header)
}

// tailReader remembers the last bytes read so that the end of a stream can be checked for a complete event.
type tailReader struct {
	r    io.Reader
	n    int64
	tail [4]byte
	tl   int
}

func (t *tailReader) Read(p []byte) (int, error) {
	n, err := t.r.Read(p)
	if n > 0 {
		t.n += int64(n)
		b := p[:n]
		if len(b) >= len(t.tail) {
			copy(t.tail[:], b[len(b)-len(t.tail):])
			t.tl = len(t.tail)
		} else {
			merged := append(t.tail[:t.tl:t.tl], b...)
			t.tl = copy(t.tail[:], merged[max(len(merged)-len(t.tail), 0):])
		}
	}
	return n, err
}

// truncated reports whether a non-empty stream ended without the blank line that dispatches an event.
func (t *tailReader) truncated() bool {
	if t.n == 0 {
		return false
	}
	b := t.tail[:t.tl]
	// Remove the terminator of the last line, then require another terminator before it.
	switch {
	case bytes.HasSuffix(b, []byte("\r\n")):
		b = b[:len(b)-2]
	case bytes.HasSuffix(b, []byte("\n")), bytes.HasSuffix(b, []byte("\r")):
		b = b[:len(b)-1]
	default:
		return true
	}
	if len(b) == 0 {
		// The tail is shorter than four bytes only when the whole stream is, so the stream was one blank line.
		return false
	}
	last := b[len(b)-1]
	return last != '\n' && last != '\r'
}
