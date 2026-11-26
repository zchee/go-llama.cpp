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

package openai_test

import (
	"errors"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

func TestResponsesNilRequest(t *testing.T) {
	lc, err := llama.New("http://127.0.0.1:1")
	if err != nil {
		t.Fatal(err)
	}
	c := openai.NewClient(lc)
	tests := map[string]struct {
		call func(*testing.T) error
		want string
	}{
		"error: Responses":            {func(t *testing.T) error { _, err := c.Responses(t.Context(), nil); return err }, "openai: nil responses request"},
		"error: ResponsesInputTokens": {func(t *testing.T) error { _, err := c.ResponsesInputTokens(t.Context(), nil); return err }, "openai: nil responses input tokens request"},
		"error: ResponsesStream": {func(t *testing.T) error {
			seq := c.ResponsesStream(t.Context(), nil)
			var first error
			for v, err := range seq {
				if v != nil || first != nil || err == nil {
					t.Fatalf("unexpected stream result: (%v, %v)", v, err)
				}
				first = err
			}
			var count int
			for v, err := range seq {
				count++
				if v != nil || !errors.Is(err, llama.ErrStreamConsumed) {
					t.Fatalf("second range: got (%v, %v), want stream consumed", v, err)
				}
			}
			if count != 1 {
				t.Fatalf("second range yielded %d events, want 1", count)
			}
			return first
		}, "openai: nil responses stream request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if err := tt.call(t); err == nil || err.Error() != tt.want {
				t.Errorf("got %v, want %q before sending", err, tt.want)
			}
		})
	}
}
