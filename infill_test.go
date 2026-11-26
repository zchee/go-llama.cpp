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
	json "encoding/json/v2"
	"slices"
	"testing"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func FuzzInfillInput(f *testing.F) {
	for _, name := range []string{"basic", "children", "stream"} {
		var req struct {
			Prefix jsontext.Value `json:"input_prefix"`
			Suffix jsontext.Value `json:"input_suffix"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/infill/"+name+".req.json"), &req); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(req.Prefix))
		f.Add([]byte(req.Suffix))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var a, b InfillInput
		if json.Unmarshal(data, &a) != nil {
			return
		}
		out, err := json.Marshal(a)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(out, &b); err != nil {
			t.Fatal(err)
		}
		if a.Text != b.Text || (a.Parts == nil) != (b.Parts == nil) || !slices.Equal(a.Parts, b.Parts) {
			t.Fatalf("input changed: %+v -> %s -> %+v", a, out, b)
		}
	})
}

func TestInfillInputRejectsNonFlatArrays(t *testing.T) {
	tests := map[string]struct{ data string }{
		"error: nested prompt":     {`[[1]]`},
		"error: multimodal prompt": {`{"prompt_string":"a"}`},
		"error: null part":         {`[null]`},
		"error: scalar number":     {`12`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var in InfillInput
			if err := json.Unmarshal([]byte(tt.data), &in); err == nil {
				t.Fatalf("accepted invalid infill input: %s", tt.data)
			}
		})
	}
}
