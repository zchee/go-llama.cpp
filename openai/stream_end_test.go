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

package openai

import (
	"bytes"
	"errors"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

// streamFrameCuts returns every prefix of a recorded stream that ends at an event boundary.
func streamFrameCuts(t testing.TB, data []byte) [][]byte {
	t.Helper()
	var cuts [][]byte
	for i := 0; i+1 < len(data); i++ {
		if data[i] == '\n' && data[i+1] == '\n' {
			cuts = append(cuts, data[:i+2])
		}
	}
	if len(cuts) == 0 || len(cuts[len(cuts)-1]) != len(data) {
		t.Fatalf("recorded stream has %d frame boundaries and %d trailing bytes", len(cuts), len(data))
	}
	return cuts
}

// checkStreamCuts replays a recorded stream cut after each complete frame: every cut must yield the
// frames before it and then llama.ErrIncompleteStream, and the whole recording must end cleanly.
// terminalYields says whether the dialect's terminal frame itself decodes into a value, as the
// named response.completed event does and the bare "[DONE]" marker does not.
func checkStreamCuts[T any](t *testing.T, fixture, path string, terminalYields bool) {
	t.Helper()
	data := jsontest.ReadFile(t, fixture)
	cuts := streamFrameCuts(t, data)
	// A caller can deliberately stop at a nonterminal frame without receiving a truncation error.
	for _, err := range llama.DecodeEndpointStream[T](bytes.NewReader(cuts[0]), 0, path, 0) {
		if err != nil {
			t.Fatal(err)
		}
		break
	}
	for i, cut := range cuts {
		var events int
		var last error
		for _, err := range llama.DecodeEndpointStream[T](bytes.NewReader(cut), 0, path, 0) {
			if err != nil {
				last = err
				break
			}
			events++
		}
		whole := i == len(cuts)-1
		want := i + 1
		if whole && !terminalYields {
			want = i
		}
		if events != want {
			t.Fatalf("cut %d yielded %d events, want %d", i, events, want)
		}
		if whole {
			if last != nil {
				t.Fatalf("whole stream ended with %v", last)
			}
		} else if !errors.Is(last, llama.ErrIncompleteStream) {
			t.Fatalf("cut %d ended with %v, want llama.ErrIncompleteStream", i, last)
		}
	}
}

func TestCompletionsStreamFixtureCuts(t *testing.T) {
	tests := map[string]struct{ fixture string }{
		"success: text deltas":        {"testdata/completions/stream.resp.sse"},
		"success: usage on the chunk": {"testdata/completions/stream_full.resp.sse"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) { checkStreamCuts[CompletionResponse](t, tt.fixture, "/v1/completions", false) })
	}
}

func TestChatCompletionsStreamFixtureCuts(t *testing.T) {
	tests := map[string]struct{ fixture string }{
		"success: message deltas": {"testdata/chat_completions/stream.resp.sse"},
		"success: reasoning":      {"testdata/chat_completions/reasoning_stream.resp.sse"},
		"success: tool calls":     {"testdata/chat_completions/tool_stream.resp.sse"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) { checkStreamCuts[ChatCompletionChunk](t, tt.fixture, "/v1/chat/completions", false) })
	}
}

func TestResponsesStreamFixtureCuts(t *testing.T) {
	tests := map[string]struct{ fixture string }{
		"success: lifecycle and deltas": {"testdata/responses/stream.resp.sse"},
		"success: reasoning":            {"testdata/responses/reasoning_stream.resp.sse"},
		"success: tool calls":           {"testdata/responses/tool_stream.resp.sse"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) { checkStreamCuts[ResponseEvent](t, tt.fixture, "/v1/responses", true) })
	}
}
