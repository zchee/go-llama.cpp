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

package llama_test

import (
	json "encoding/json/v2"
	"reflect"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/anthropic"
)

func TestBorrowedViews(t *testing.T) {
	tests := map[string]struct{ check func(*testing.T) }{
		"success: prompt tokens":           {borrowedView(llama.Prompt.Tokens, `[1]`, `[2]`)},
		"success: mixed prompt parts":      {borrowedView(llama.Prompt.Parts, `[1,"a"]`, `[2,"b"]`)},
		"success: batched prompts":         {borrowedView(llama.Prompt.Batch, `["a"]`, `["b"]`)},
		"success: logit bias entries":      {borrowedView(llama.LogitBias.Entries, `[[1,0.5]]`, `[[2,1]]`)},
		"success: string list":             {borrowedView(llama.StringOrSlice.Strings, `["a"]`, `["b"]`)},
		"success: message content parts":   {borrowedView(llama.MessageContent.Parts, `[{"type":"text","text":"a"}]`, `[{"type":"text","text":"b"}]`)},
		"success: tokenization parts":      {borrowedView(llama.TokenizeContent.Parts, `[1]`, `[2]`)},
		"success: token piece bytes":       {borrowedView(llama.TokenPiece.Bytes, `[1]`, `[2]`)},
		"success: embedding input batch":   {borrowedView(llama.EmbeddingInput.Batch, `["a"]`, `["b"]`)},
		"success: embedding vector values": {borrowedView(llama.EmbeddingVector.Values, `[1]`, `[2]`)},
		"success: unknown tool choice":     {borrowedView(llama.ToolChoice.Unknown, `{"type":"a"}`, `{"type":"b"}`)},
		"success: message content blocks":  {borrowedView(anthropic.Content.Blocks, `[{"type":"text","text":"a"}]`, `[{"type":"text","text":"b"}]`)},
		"success: raw message content":     {borrowedView(anthropic.Content.Raw, `{"a":1}`, `{"b":2}`)},
	}
	for name, tt := range tests {
		t.Run(name, tt.check)
	}
}

func borrowedView[T any, S ~[]E, E any](access func(T) S, first, second string) func(*testing.T) {
	return func(t *testing.T) {
		t.Helper()
		var value T
		if err := json.Unmarshal([]byte(first), &value); err != nil {
			t.Fatal(err)
		}
		old := access(value)
		if len(old) == 0 || &old[0] != &access(value)[0] {
			t.Fatal("accessor did not share the stored slice")
		}
		if err := json.Unmarshal([]byte(second), &value); err != nil {
			t.Fatal(err)
		}
		var original T
		if err := json.Unmarshal([]byte(first), &original); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(old, access(original)) {
			t.Error("decoding the next value mutated the caller's earlier view")
		}
		if next := access(value); len(next) == 0 || &next[0] == &old[0] {
			t.Error("decoding the next value reused a slice still held by the caller")
		}
	}
}
