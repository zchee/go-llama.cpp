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
	json "encoding/json/v2"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
)

func TestToolChoiceModeWrapper(t *testing.T) {
	tests := map[string]struct {
		mode     string
		wantJSON string
	}{
		"success: auto":     {mode: "auto", wantJSON: `"auto"`},
		"success: none":     {mode: "none", wantJSON: `"none"`},
		"success: required": {mode: "required", wantJSON: `"required"`},
		"success: empty":    {mode: "", wantJSON: `""`},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got := ToolChoiceMode(tt.mode)
			want := llama.ToolChoiceMode(tt.mode)
			if diff := gocmp.Diff(want, got); diff != "" {
				t.Errorf("ToolChoiceMode(%q) differs from llama.ToolChoiceMode (-want +got):\n%s", tt.mode, diff)
			}

			gotJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("Marshal(openai.ToolChoiceMode): %v", err)
			}
			wantJSON, err := json.Marshal(want)
			if err != nil {
				t.Fatalf("Marshal(llama.ToolChoiceMode): %v", err)
			}
			if !bytes.Equal(gotJSON, wantJSON) {
				t.Errorf("JSON = %s, want %s", gotJSON, wantJSON)
			}
			if string(gotJSON) != tt.wantJSON {
				t.Errorf("JSON = %s, want %s", gotJSON, tt.wantJSON)
			}
		})
	}
}
