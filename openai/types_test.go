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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestNewClient(t *testing.T) {
	lc, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
	if got := NewClient(lc).Llama(); got != lc {
		t.Errorf("NewClient(c).Llama() = %v, want the client it was built from", got)
	}
}

func TestMessageContentJSON(t *testing.T) {
	tests := map[string]struct {
		content  MessageContent
		json     string
		wantKind ContentKind
	}{
		"success: text": {content: TextContent("Hello"), json: `"Hello"`, wantKind: ContentKindText},
		"success: parts": {
			content:  PartsContent(TextPart("What is this?"), ImageURLPart("data:image/png;base64,iVBORw==")),
			json:     `[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw=="}}]`,
			wantKind: ContentKindParts,
		},
		"success: empty parts": {content: PartsContent(), json: `[]`, wantKind: ContentKindParts},
		"success: null":        {content: NullContent(), json: `null`, wantKind: ContentKindNull},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := json.Marshal(tt.content)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.json {
				t.Errorf("Marshal = %s, want %s", got, tt.json)
			}
			var back MessageContent
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatal(err)
			}
			if back.Kind() != tt.wantKind {
				t.Errorf("Kind = %d, want %d", back.Kind(), tt.wantKind)
			}
			if diff := gocmp.Diff(tt.content, back); diff != "" {
				t.Errorf("round trip (-want +got):\n%s", diff)
			}
		})
	}
	var c MessageContent
	if err := json.Unmarshal([]byte(`3`), &c); err == nil {
		t.Errorf("content accepted a number")
	}
	if TextContent("x").Text() != "x" || len(PartsContent(TextPart("a")).Parts()) != 1 || !(MessageContent{}).IsZero() {
		t.Errorf("accessors do not return the built content")
	}
}

func TestToolChoiceAliases(t *testing.T) {
	data := jsontest.ReadFile(t, "../testdata/apply-template/allowed_tools.req.json")
	got := jsontest.CheckWire[chatShape](t, data)
	jsontest.CheckFallbacks[AllowedToolChoice](t)
	want := ToolChoiceAllowedTools("required", "weather")
	if diff := gocmp.Diff(want, got.ToolChoice); diff != "" {
		t.Errorf("allowlist differs (-want +got):\n%s", diff)
	}
	if len(got.Tools) != 1 || got.Tools[0].DeferLoading == nil || !*got.Tools[0].DeferLoading {
		t.Fatalf("expected the tool-level defer_loading flag, got %+v", got.Tools)
	}
	function := got.Tools[0].Function
	if function == nil || function.DeferLoading == nil || *function.DeferLoading {
		t.Errorf("expected explicit false on the function-level defer_loading flag, got %+v", function)
	}
}

// chatShape is the subset of a chat request that the shared types cover; the endpoint request types embed them
// the same way.
type chatShape struct {
	Messages       []Message                 `json:"messages"`
	Tools          []Tool                    `json:"tools,omitzero"`
	ToolChoice     ToolChoice                `json:"tool_choice,omitzero"`
	ResponseFormat *ResponseFormat           `json:"response_format,omitzero"`
	StreamOptions  *StreamOptions            `json:"stream_options,omitzero"`
	Usage          *Usage                    `json:"usage,omitzero"`
	Extra          map[string]jsontext.Value `json:",embed"`
}

func TestSharedTypesKeepUnknownMembers(t *testing.T) {
	data, err := json.Marshal(chatShape{
		Messages: []Message{
			{Role: "system", Content: TextContent("be brief")},
			{Role: "user", Content: PartsContent(TextPart("hi"), ContentPart{Type: "input_audio", InputAudio: &MediaInput{Data: new("UklGRg=="), Format: new("wav")}})},
			{
				Role:      "assistant",
				Content:   NullContent(),
				ToolCalls: []ToolCall{{ID: new("call_1"), Type: new("function"), Function: &FunctionCall{Name: new("f"), Arguments: new(`{"a":1}`)}}},
			},
			{Role: "tool", ToolCallID: new("call_1"), Content: TextContent("42")},
		},
		Tools:      []Tool{{Type: "function", Function: &FunctionDefinition{Name: "f", Parameters: jsontext.Value(`{"type":"object"}`)}}},
		ToolChoice: ToolChoiceFunction("f"),
		ResponseFormat: &ResponseFormat{
			Type:       "json_schema",
			JSONSchema: &JSONSchemaFormat{Name: "s", Schema: jsontext.Value(`{"type":"object"}`), Strict: new(true)},
		},
		StreamOptions: &StreamOptions{IncludeUsage: new(true)},
		Usage:         &Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3, PromptTokensDetails: &llama.PromptTokensDetails{}},
	}, json.Deterministic(true))
	if err != nil {
		t.Fatal(err)
	}
	jsontest.RoundTrip[chatShape](t, data)
	jsontest.InjectUnknown[chatShape](t, data, jsontest.SkipObjects(
		// Raw JSON schemas are kept as is, so an injected member survives without exercising a fallback.
		"/tools/0/function/parameters", "/response_format/json_schema/schema",
	))
	jsontest.NoUnknown[chatShape](t, data, nil)
	jsontest.CheckFallbacks[chatShape](t)
	jsontest.CheckFallbacks[NamedToolChoice](t)
	jsontest.CheckFallbacks[ContentPart](t)
}

// TestNoUnknownSeesInsideSumTypes checks that the unknown-member gate reaches the content parts of a
// MessageContent and the object of a ToolChoice, which are kept in unexported fields.
func TestNoUnknownSeesInsideSumTypes(t *testing.T) {
	tests := map[string]struct {
		check func(tb testing.TB)
		want  string
	}{
		"error: image of a content part": {
			check: func(tb testing.TB) {
				jsontest.NoUnknown[Message](tb, []byte(`{"role":"user","content":[{"type":"image_url","image_url":{"url":"u","x":1}}]}`), nil)
			},
			want: "/content/0/image_url: x",
		},
		"error: named tool choice": {
			check: func(tb testing.TB) {
				jsontest.NoUnknown[ToolChoice](tb, []byte(`{"type":"function","function":{"name":"f","later":2}}`), nil)
			},
			want: "/function: later",
		},
		"success: nothing unknown": {
			check: func(tb testing.TB) {
				jsontest.NoUnknown[Message](tb, []byte(`{"role":"user","content":[{"type":"text","text":"x"}]}`), nil)
			},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			msgs, _ := jsontest.Failures(t, tt.check)
			if tt.want == "" {
				if len(msgs) != 0 {
					t.Errorf("unexpected failures: %v", msgs)
				}
				return
			}
			if len(msgs) != 1 || !strings.Contains(msgs[0], tt.want) {
				t.Errorf("failures = %v, want one naming %q", msgs, tt.want)
			}
		})
	}
}
