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

package anthropic

import (
	"bytes"
	"errors"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

// TestMessagesStreamFixtureCuts replays recorded /v1/messages streams cut after each complete
// frame: every cut must yield the events before it and then llama.ErrIncompleteStream, because the
// stream is complete only after the message_stop event; the whole recording must end cleanly.
func TestMessagesStreamFixtureCuts(t *testing.T) {
	tests := map[string]struct{ fixture string }{
		"success: text deltas": {"testdata/messages/stream.resp.sse"},
		"success: thinking":    {"testdata/messages/thinking_stream.resp.sse"},
		"success: tool use":    {"testdata/messages/tool_stream.resp.sse"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, tt.fixture)
			cuts := jsontest.StreamFrameCuts(t, data)
			for _, err := range llama.DecodeEndpointStream[MessageEvent](bytes.NewReader(cuts[0]), 0, "/v1/messages", 0) {
				if err != nil {
					t.Fatal(err)
				}
				break
			}
			for i, cut := range cuts {
				var events int
				var last error
				for _, err := range llama.DecodeEndpointStream[MessageEvent](bytes.NewReader(cut), 0, "/v1/messages", 0) {
					if err != nil {
						last = err
						break
					}
					events++
				}
				if events != i+1 {
					t.Fatalf("cut %d yielded %d events, want %d", i, events, i+1)
				}
				if whole := i == len(cuts)-1; whole {
					if last != nil {
						t.Fatalf("whole stream ended with %v", last)
					}
				} else if !errors.Is(last, llama.ErrIncompleteStream) {
					t.Fatalf("cut %d ended with %v, want llama.ErrIncompleteStream", i, last)
				}
			}
		})
	}
}
