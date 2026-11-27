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
	"reflect"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestToolChoiceJSON(t *testing.T) {
	tests := map[string]struct {
		choice ToolChoice
		json   string
	}{
		"success: mode":     {choice: ToolChoiceMode("required"), json: `"required"`},
		"success: function": {choice: ToolChoiceFunction("get_weather"), json: `{"type":"function","function":{"name":"get_weather"}}`},
		"success: allowlist": {
			choice: ToolChoiceAllowedTools("required", "get_weather"),
			json:   `{"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[{"type":"function","function":{"name":"get_weather"}}]}}`,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := json.Marshal(tt.choice)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.json {
				t.Errorf("Marshal = %s, want %s", got, tt.json)
			}
			var back ToolChoice
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tt.choice, back); diff != "" {
				t.Errorf("round trip (-want +got):\n%s", diff)
			}
		})
	}
	var c ToolChoice
	if err := json.Unmarshal([]byte(`[]`), &c); err == nil {
		t.Errorf("tool_choice accepted an array")
	}
	if ToolChoiceMode("auto").Mode() != "auto" || ToolChoiceFunction("f").Function().Function.Name != "f" || !(ToolChoice{}).IsZero() {
		t.Errorf("accessors do not return the built choice")
	}
	allowed := ToolChoiceAllowedTools("auto", "f")
	if allowed.IsZero() || allowed.Mode() != "" || allowed.Function() != nil || allowed.AllowedTools().AllowedTools.Tools[0].Function.Name != "f" {
		t.Errorf("accessors do not return the built allowlist: %+v", allowed)
	}
}

func TestToolChoiceUnknownDiscriminator(t *testing.T) {
	tests := map[string]struct {
		input string
	}{
		"success: unknown object with a function":                 {input: `{"type":"future","function":{"name":"f","future_nested":true},"config":{"enabled":true}}`},
		"success: unknown object without a function":              {input: `{"type":"future","config":{"enabled":true}}`},
		"success: unknown object with a different function shape": {input: `{"type":"future","function":false}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var got ToolChoice
			if err := json.Unmarshal([]byte(tt.input), &got); err != nil {
				t.Fatal(err)
			}
			if got.AllowedTools() != nil || got.Function() != nil || got.Mode() != "" || got.IsZero() || string(got.Unknown()) != tt.input {
				t.Errorf("unknown discriminator reported as a known or unset choice: %+v", got)
			}
			encoded, err := json.Marshal(got, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tt.input, string(encoded)); diff != "" {
				t.Errorf("unknown tool choice changed (-want +got):\n%s", diff)
			}
		})
	}
}

type chatToolsWireRequest struct {
	Messages             []Message                 `json:"messages"`
	Tools                []Tool                    `json:"tools"`
	ToolChoice           ToolChoice                `json:"tool_choice"`
	ResponseFormat       *ResponseFormat           `json:"response_format,omitzero"`
	ContinueFinalMessage jsontext.Value            `json:"continue_final_message,omitzero"`
	Extra                map[string]jsontext.Value `json:",embed"`
}

func TestChatToolsWire(t *testing.T) {
	tests := map[string]struct {
		stem string
		want ToolChoice
	}{
		"success: named function":                 {"named_tool", ToolChoiceFunction("weather")},
		"success: mode with response schema":      {"tool_mode", ToolChoiceMode("none")},
		"success: allowlist and deferred loading": {"allowed_tools", ToolChoiceAllowedTools("required", "weather")},
	}
	jsontest.CheckFallbacks[chatToolsWireRequest](t)
	jsontest.CheckFallbacks[NamedToolChoice](t)
	jsontest.CheckFallbacks[AllowedToolChoice](t)
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/apply-template/"+tt.stem+".req.json")
			got := jsontest.CheckWire[chatToolsWireRequest](t, data)
			if diff := gocmp.Diff(tt.want, got.ToolChoice); diff != "" {
				t.Errorf("tool choice differs (-want +got):\n%s", diff)
			}
		})
	}
}

func FuzzToolChoiceJSON(f *testing.F) {
	for _, stem := range []string{"named_tool", "tool_mode", "allowed_tools"} {
		var request struct {
			ToolChoice jsontext.Value `json:"tool_choice"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/apply-template/"+stem+".req.json"), &request); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(request.ToolChoice))
	}
	f.Add([]byte(`{"type":"allowed_tools","allowed_tools":{}}`))
	f.Add([]byte(`{"type":"future","config":{"enabled":true}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var before, after ToolChoice
		if err := json.Unmarshal(data, &before, jsonOptions); err != nil {
			return
		}
		encoded, err := json.Marshal(before, jsonOptions)
		if err != nil {
			t.Fatalf("encoding tool choice: %v (input %q)", err, data)
		}
		if err := json.Unmarshal(encoded, &after, jsonOptions); err != nil {
			t.Fatalf("decoding re-encoded tool choice: %v (input %q, encoded %q)", err, data, encoded)
		}
		// ToolChoice.Equal compares the bytes of unknown values. Ignore formatting, not contents, here.
		type choice ToolChoice
		equalRawJSON := gocmp.FilterPath(func(p gocmp.Path) bool {
			return p.Last().Type() == reflect.TypeFor[jsontext.Value]()
		}, gocmp.Comparer(func(x, y jsontext.Value) bool {
			x, y = x.Clone(), y.Clone()
			opts := json.JoinOptions(jsontext.CanonicalizeRawInts(false), jsontext.CanonicalizeRawFloats(false))
			return x.Canonicalize(opts) == nil && y.Canonicalize(opts) == nil && x.String() == y.String()
		}))
		if diff := gocmp.Diff(choice(before), choice(after), gocmp.AllowUnexported(choice{}), equalRawJSON); diff != "" {
			t.Fatalf("tool choice changed on round trip (-before +after):\n%s\ninput %q, re-encoded %q", diff, data, encoded)
		}
	})
}
