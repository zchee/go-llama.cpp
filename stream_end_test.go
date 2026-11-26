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
	"encoding/json/jsontext"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestStreamEndNativeFixtureCuts(t *testing.T) {
	tests := map[string]struct {
		fixture  string
		expected int
		fields   []string
	}{
		"success: single completion":   {"completion/stream", 1, nil},
		"success: single infill":       {"infill/stream", 1, nil},
		"success: sibling completions": {"completion/children_stream", 2, nil},
		"success: projected batch":     {"completion/projected_stream", 2, []string{"content"}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/"+tt.fixture+".resp.sse")
			endpoint, _, _ := strings.Cut(tt.fixture, "/")
			if !bytes.HasSuffix(data, []byte("\n\n")) {
				t.Fatal("fixture ends inside a frame")
			}
			frames, end := 0, 0
			for {
				seq := DecodeEndpointStream[CompletionResponse](bytes.NewReader(data[:end]), 0, "/"+endpoint, tt.expected, tt.fields...)
				var events, failures int
				for v, err := range seq {
					if err != nil {
						if v != nil || !errors.Is(err, ErrIncompleteStream) {
							t.Fatalf("cut after %d events: %v, %v", frames, v, err)
						}
						failures++
					} else {
						events++
					}
				}
				if events != frames || failures > 1 || (failures == 0) != (end == len(data)) {
					t.Fatalf("cut after %d/%d bytes: got %d events and %d errors; expected %d events", end, len(data), events, failures, frames)
				}
				if end == len(data) {
					break
				}
				end += bytes.Index(data[end:], []byte("\n\n")) + 2
				frames++
				if frames == 1 {
					for _, err := range DecodeEndpointStream[CompletionResponse](bytes.NewReader(data[:end]), 0, "/"+endpoint, tt.expected, tt.fields...) {
						if err != nil {
							t.Fatal(err)
						}
						break
					}
				}
			}
		})
	}
}

func TestProjectedStreamNullDoesNotFinishResult(t *testing.T) {
	data := string(jsontest.ReadFile(t, "testdata/completion/projected_stream.resp.sse"))
	begin, frames, ok := strings.Cut(data, "\n\n")
	if !ok || begin != "data: null" {
		t.Fatalf("expected a recorded null frame, got %q", begin)
	}
	partial, frames, _ := strings.Cut(frames, "\n\n")
	final, _, _ := strings.Cut(frames, "\n\n")
	// One result finishes; the second starts but never finishes.
	prefix := partial + "\n\n" + final + "\n\n"
	tests := map[string]struct{ begin string }{
		"error: recorded null": {begin},
		"error: padded null":   {strings.Replace(begin, "null", "\t null \t", 1)},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var last error
			for _, err := range DecodeEndpointStream[CompletionResponse](strings.NewReader(prefix+tt.begin+"\n\n"), 0, "/completion", 2, "content") {
				last = err
			}
			if !errors.Is(last, ErrIncompleteStream) {
				t.Errorf("stream ended with %v, want ErrIncompleteStream", last)
			}
		})
	}
}

func TestStreamEndMarkers(t *testing.T) {
	tests := map[string]struct {
		path, stream   string
		wantIncomplete bool
	}{
		"error: OpenAI alias with query":          {"/chat/completions?client=x", "", true},
		"error: DONE is not response.completed":   {"/responses", "data: [DONE]\n\n", true},
		"error: missing index without projection": {"/completion", "data: {\"stop\":true}\n\n", true},
		"error: missing stop without projection":  {"/completion", "data: {\"index\":0}\n\n", true},
		"error: begin signal is not a final":      {"/infill", "data: null\n\n", true},
		"success: model feed shutdown":            {"/models/sse", "data: {\"model\":\"m\",\"event\":\"models_reload\"}\n\n", false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var last error
			for _, err := range DecodeEndpointStream[jsontext.Value](strings.NewReader(tt.stream), 0, tt.path, 0) {
				last = err
			}
			if (last != nil) != tt.wantIncomplete || (last != nil && !errors.Is(last, ErrIncompleteStream)) {
				t.Errorf("got %v, want incomplete=%t", last, tt.wantIncomplete)
			}
		})
	}
}

func TestStreamEndUnexpectedTransportEOF(t *testing.T) {
	data := jsontest.ReadFile(t, "openai/testdata/completions/stream.resp.sse")
	first, _, _ := bytes.Cut(data, []byte("\n\n"))
	r := io.MultiReader(bytes.NewReader(data[:len(first)+2]), iotest.ErrReader(io.ErrUnexpectedEOF))
	var last error
	for _, err := range DecodeEndpointStream[jsontext.Value](r, 0, "/v1/completions", 0) {
		last = err
	}
	if !errors.Is(last, ErrIncompleteStream) || !errors.Is(last, io.ErrUnexpectedEOF) {
		t.Fatalf("lost termination or transport error: %v", last)
	}
}

func TestNativeResultCount(t *testing.T) {
	c, err := New("http://localhost:8080")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		body any
		want int
	}{
		"success: batch times siblings":            {&CompletionRequest{Prompt: BatchPrompt(TextPrompt("a"), TextPrompt("b")), NCmpl: new(3)}, 6},
		"success: infill n alias":                  {&InfillRequest{Extra: map[string]jsontext.Value{"n": jsontext.Value("2")}}, 2},
		"success: raw token prompt":                {map[string]any{"prompt": []int{1, 2}, "n": 3}, 3},
		"success: raw batch and n_cmpl precedence": {map[string]any{"prompt": []string{"a", "b"}, "n": 1, "n_cmpl": 2}, 4},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			r, err := c.NewRequest(t.Context(), http.MethodPost, "/completion", tt.body)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Body.Close()
			if got, _ := nativeStreamRequest(tt.body, r); got != tt.want {
				t.Errorf("got %d expected finals, want %d", got, tt.want)
			}
		})
	}
}
