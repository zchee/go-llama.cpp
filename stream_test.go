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
	"encoding/json/jsontext"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"

	gocmp "github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// chunk is a minimal event payload used to check decoding.
type chunk struct {
	Type    string                    `json:"type,omitzero"`
	Content string                    `json:"content,omitzero"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

type streamResult struct {
	events []chunk
	err    error
}

func collect(seq func(func(*chunk, error) bool)) streamResult {
	var r streamResult
	for v, err := range seq {
		if err != nil {
			r.err = err
			break
		}
		r.events = append(r.events, *v)
	}
	return r
}

// The streams below follow format_oai_sse, format_oai_resp_sse and format_anthropic_sse in
// tools/server/server-common.cpp and the stream loop in tools/server/server-context.cpp of llama.cpp v1.1.0.
func TestReadStreamDialects(t *testing.T) {
	tests := map[string]struct {
		stream      string
		maxEvent    int
		want        []chunk
		wantAPIErr  *APIError
		wantErrIs   error
		wantErrText string
	}{
		"success: data lines ended by [DONE]": {
			stream: "data: {\"content\":\"a\"}\n\ndata: {\"content\":\"b\"}\n\ndata: [DONE]\n\n",
			want:   []chunk{{Content: "a"}, {Content: "b"}},
		},
		"success: nothing is read after [DONE]": {
			stream: "data: {\"content\":\"a\"}\n\ndata: [DONE]\n\ndata: {broken",
			want:   []chunk{{Content: "a"}},
		},
		"success: data lines ended by EOF": {
			stream: "data: {\"content\":\"a\"}\n\ndata: {\"content\":\"b\",\"stop\":true}\n\n",
			want:   []chunk{{Content: "a"}, {Content: "b", Extra: map[string]jsontext.Value{"stop": jsontext.Value("true")}}},
		},
		"success: event and data pairs": {
			stream: "event: response.created\ndata: {\"type\":\"response.created\"}\n\n" +
				"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"content\":\"x\"}\n\n",
			want: []chunk{{Type: "response.created"}, {Type: "response.output_text.delta", Content: "x"}},
		},
		"success: comment pings are skipped": {
			stream: ":\n\ndata: {\"content\":\"a\"}\n\n:\n\n:\n\ndata: {\"content\":\"b\"}\n\n",
			want:   []chunk{{Content: "a"}, {Content: "b"}},
		},
		"success: CRLF line endings": {
			stream: "data: {\"content\":\"a\"}\r\n\r\ndata: [DONE]\r\n\r\n",
			want:   []chunk{{Content: "a"}},
		},
		"success: empty body": {
			stream: "",
			want:   nil,
		},
		"error: in-stream error of the data dialects": {
			stream:     "data: {\"content\":\"a\"}\n\ndata: {\"error\":{\"code\":500,\"message\":\"boom\",\"type\":\"server_error\"}}\n\n",
			want:       []chunk{{Content: "a"}},
			wantAPIErr: &APIError{StatusCode: http.StatusOK, Code: 500, Message: "boom", Type: ErrorTypeServer},
		},
		"error: in-stream context size error": {
			stream:     "data: {\"error\":{\"code\":400,\"message\":\"too long\",\"type\":\"exceed_context_size_error\",\"n_prompt_tokens\":600,\"n_ctx\":512}}\n\n",
			wantAPIErr: &APIError{StatusCode: http.StatusOK, Code: 400, Message: "too long", Type: ErrorTypeExceedContextSize, NPromptTokens: 600, NCtx: 512},
		},
		"error: named error event with the bare object": {
			stream:     "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: error\ndata: {\"code\":500,\"message\":\"boom\",\"type\":\"server_error\"}\n\n",
			want:       []chunk{{Type: "message_start"}},
			wantAPIErr: &APIError{StatusCode: http.StatusOK, Code: 500, Message: "boom", Type: ErrorTypeServer},
		},
		"error: named error event with an envelope": {
			stream:     "event: error\ndata: {\"error\":{\"code\":503,\"message\":\"down\",\"type\":\"unavailable_error\"}}\n\n",
			wantAPIErr: &APIError{StatusCode: http.StatusOK, Code: 503, Message: "down", Type: ErrorTypeUnavailable},
		},
		"error: named error event that is not JSON": {
			stream:     "event: error\ndata: upstream gone\n\n",
			wantAPIErr: &APIError{StatusCode: http.StatusOK, Message: "upstream gone"},
		},
		"success: an error member that is not an object is data": {
			stream: "data: {\"content\":\"a\",\"error\":null}\n\n",
			want:   []chunk{{Content: "a", Extra: map[string]jsontext.Value{"error": jsontext.Value("null")}}},
		},
		"error: event larger than the limit": {
			stream:      "data: {\"content\":\"" + strings.Repeat("x", 200) + "\"}\n\n",
			maxEvent:    64,
			wantErrText: "reading event stream",
		},
		"error: body ends inside an event": {
			stream:    "data: {\"content\":\"a\"}\n\ndata: {\"content\":\"b\"}\n",
			want:      []chunk{{Content: "a"}},
			wantErrIs: io.ErrUnexpectedEOF,
		},
		"error: body ends inside a line": {
			stream:    "data: {\"content\":\"a\"}\n\ndata: {\"cont",
			want:      []chunk{{Content: "a"}},
			wantErrIs: io.ErrUnexpectedEOF,
		},
		"error: broken JSON": {
			stream:      "data: {\"content\":\"a\"}\n\ndata: {\"content\":\n\n",
			want:        []chunk{{Content: "a"}},
			wantErrText: "decoding event",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			// One byte per read exercises the tail bookkeeping across reads.
			for _, r := range map[string]io.Reader{
				"whole":    strings.NewReader(tt.stream),
				"one byte": iotest.OneByteReader(strings.NewReader(tt.stream)),
			} {
				got := collect(readStream[chunk](r, tt.maxEvent, http.StatusOK, http.Header{"X-Test": {"1"}}))
				if diff := gocmp.Diff(tt.want, got.events, cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("events mismatch (-want +got):\n%s", diff)
				}
				switch {
				case tt.wantAPIErr != nil:
					apiErr, ok := errors.AsType[*APIError](got.err)
					if !ok {
						t.Fatalf("error = %v, want *APIError", got.err)
					}
					if diff := gocmp.Diff(tt.wantAPIErr, apiErr, cmpopts.IgnoreFields(APIError{}, "Raw", "Header")); diff != "" {
						t.Errorf("APIError mismatch (-want +got):\n%s", diff)
					}
					if apiErr.Header.Get("X-Test") != "1" {
						t.Errorf("APIError.Header is not the stream response header")
					}
				case tt.wantErrIs != nil:
					if !errors.Is(got.err, tt.wantErrIs) {
						t.Errorf("error = %v, want %v", got.err, tt.wantErrIs)
					}
				case tt.wantErrText != "":
					if got.err == nil || !strings.Contains(got.err.Error(), tt.wantErrText) {
						t.Errorf("error = %v, want one containing %q", got.err, tt.wantErrText)
					}
				default:
					if got.err != nil {
						t.Errorf("unexpected error: %v", got.err)
					}
				}
			}
		})
	}
}

func TestReadStreamStopsOnBreak(t *testing.T) {
	r := &countingReader{r: strings.NewReader(strings.Repeat("data: {\"content\":\"a\"}\n\n", 10000))}
	n := 0
	for _, err := range readStream[chunk](r, 0, http.StatusOK, nil) {
		if err != nil {
			t.Fatal(err)
		}
		n++
		if n == 2 {
			break
		}
	}
	if n != 2 {
		t.Fatalf("got %d events, want 2", n)
	}
	if r.n > 64<<10 {
		t.Errorf("read %d bytes after break; the parser kept reading", r.n)
	}
}

func TestDecodeStream(t *testing.T) {
	got := collect(DecodeStream[chunk](strings.NewReader("data: {\"error\":{\"code\":500,\"message\":\"m\",\"type\":\"server_error\"}}\n\n"), 0))
	apiErr, ok := errors.AsType[*APIError](got.err)
	if !ok || apiErr.StatusCode != 0 || apiErr.Type != ErrorTypeServer {
		t.Errorf("DecodeStream error = %#v, want an *APIError with zero StatusCode", got.err)
	}
}

func TestCheckEventStream(t *testing.T) {
	tests := map[string]struct {
		contentType string
		wantErr     bool
	}{
		"success: plain":           {contentType: "text/event-stream"},
		"success: with parameters": {contentType: "text/event-stream; charset=utf-8"},
		"error: JSON body":         {contentType: "application/json; charset=utf-8", wantErr: true},
		"error: missing":           {contentType: "", wantErr: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			resp := &http.Response{Header: http.Header{}}
			if tt.contentType != "" {
				resp.Header.Set("Content-Type", tt.contentType)
			}
			if err := checkEventStream(resp); (err != nil) != tt.wantErr {
				t.Errorf("checkEventStream(%q) = %v, want error %t", tt.contentType, err, tt.wantErr)
			}
		})
	}
}

// TestStreamSingleUse checks that a second range neither sends nor reuses the first one: the first range fails to
// connect to a closed port, the second yields ErrStreamConsumed without dialing.
func TestStreamSingleUse(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	c, err := New("http://" + addr)
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()

	seq := Stream[chunk](t.Context(), c, http.MethodPost, "/completion", map[string]any{"stream": true})
	first := collect(seq)
	if first.err == nil || errors.Is(first.err, ErrStreamConsumed) {
		t.Fatalf("first range error = %v, want a connection error", first.err)
	}
	var errs []error
	for v, err := range seq {
		if v != nil {
			t.Errorf("second range yielded a value")
		}
		errs = append(errs, err)
	}
	if len(errs) != 1 || !errors.Is(errs[0], ErrStreamConsumed) {
		t.Errorf("second range yielded %v, want exactly [ErrStreamConsumed]", errs)
	}

	bad := Stream[chunk](t.Context(), c, http.MethodPost, "/completion", map[string]any{"f": func() {}})
	if got := collect(bad); got.err == nil || !strings.Contains(got.err.Error(), "encoding request body") {
		t.Errorf("unencodable body error = %v", got.err)
	}
}

type countingReader struct {
	r io.Reader
	n int
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}
