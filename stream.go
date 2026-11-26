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
	"sync/atomic"

	sse "github.com/tmaxmax/go-sse"
)

// ErrStreamConsumed is yielded when a stream returned by Stream is ranged over a second time. A stream sends its
// request when the range loop starts and can be consumed only once.
var ErrStreamConsumed = errors.New("llama: stream already consumed")

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
func Stream[T any](ctx context.Context, c *Client, method, path string, body any, opts ...RequestOption) iter.Seq2[*T, error] {
	var used atomic.Bool
	return func(yield func(*T, error) bool) {
		if used.Swap(true) {
			yield(nil, ErrStreamConsumed)
			return
		}
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
		for v, err := range readStream[T](resp.Body, c.maxEventSize, resp.StatusCode, resp.Header) {
			if err != nil && ctx.Err() != nil {
				err = fmt.Errorf("llama: stream interrupted: %w", context.Cause(ctx))
			}
			err = c.RedactError(err)
			if !yield(v, err) || err != nil {
				return
			}
		}
	}
}

// DecodeStream decodes a server-sent event stream read from r into T values with the rules of Stream, for
// example to replay a recorded stream. maxEventSize limits the size of one event; zero or less selects
// DefaultMaxEventSize. Errors carried by events are yielded as *APIError with a zero StatusCode.
func DecodeStream[T any](r io.Reader, maxEventSize int) iter.Seq2[*T, error] {
	return readStream[T](r, maxEventSize, 0, nil)
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
func readStream[T any](r io.Reader, maxEventSize, status int, header http.Header) iter.Seq2[*T, error] {
	if maxEventSize <= 0 {
		maxEventSize = DefaultMaxEventSize
	}
	return func(yield func(*T, error) bool) {
		tr := &tailReader{r: r}
		for ev, err := range sse.Read(tr, &sse.ReadConfig{MaxEventSize: maxEventSize}) {
			if err != nil {
				yield(nil, fmt.Errorf("llama: reading event stream: %w", err))
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
			if !yield(v, nil) {
				return
			}
		}
		if tr.truncated() {
			yield(nil, fmt.Errorf("llama: event stream ended inside an event: %w", io.ErrUnexpectedEOF))
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
