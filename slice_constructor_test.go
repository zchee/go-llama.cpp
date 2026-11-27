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
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/anthropic"
	"github.com/zchee/go-llama.cpp/openai"
)

func TestSliceConstructors(t *testing.T) {
	tests := map[string]struct{ check func(*testing.T) }{
		"success: message parts":      {sliceConstructor(llama.PartsContent, llama.TextContentPart("a"), `[{"type":"text","text":"a"}]`)},
		"success: tokenization parts": {sliceConstructor(llama.TokenizeParts, llama.TokenPart(7), `[7]`)},
		"success: token piece bytes":  {sliceConstructor(func(b ...byte) llama.TokenPiece { return llama.TokenPieceBytes(b) }, byte(7), `[7]`)},
		"success: string list":        {sliceConstructor(llama.StringSlice, "a", `["a"]`)},
		"success: response items":     {sliceConstructor(openai.ResponseInputItems, openai.ResponseItem{Value: &openai.ResponseMessage{Role: "user"}}, `[{"role":"user"}]`)},
		"success: response parts":     {sliceConstructor(openai.ResponseContentParts, openai.ResponsePart{Value: &openai.ResponseTextPart{Type: "input_text", Text: "a"}}, `[{"type":"input_text","text":"a"}]`)},
		"success: content blocks":     {sliceConstructor(anthropic.BlocksContent, anthropic.ContentBlock{Type: "text", Text: new("a")}, `[{"type":"text","text":"a"}]`)},
	}
	for name, tt := range tests {
		t.Run(name, tt.check)
	}
}

func sliceConstructor[E, V any](build func(...E) V, element E, want string) func(*testing.T) {
	return func(t *testing.T) {
		check := func(v V, want string) {
			t.Helper()
			got, err := json.Marshal(v, json.FormatNilSliceAsNull(true))
			if err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(want, string(got)); diff != "" {
				t.Errorf("wire JSON mismatch (-want +got):\n%s", diff)
			}
		}
		check(build(), "[]")
		source := []E{element}
		value := build(source...)
		clear(source)
		check(value, want)
	}
}
