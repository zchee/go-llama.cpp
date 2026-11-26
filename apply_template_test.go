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
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestApplyTemplateWire(t *testing.T) {
	user := Message{Role: "user", Content: TextContent("Say hi")}
	assistant := Message{Role: "assistant", Content: TextContent("Hi")}
	continued := "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\nHi"
	tool := Tool{
		Type:     "function",
		Function: &FunctionDefinition{Name: "weather", Parameters: jsontext.Value(`{"type":"object"}`)},
	}
	tools := []Tool{tool}
	deferredTool := tool
	deferredTool.DeferLoading = new(true)
	deferredTool.Function = new(*tool.Function)
	deferredTool.Function.DeferLoading = new(false)
	weather := Message{Role: "user", Content: TextContent("Weather?")}
	weatherPrompt := "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n"
	tests := map[string]struct {
		request ApplyTemplateRequest
		prompt  string
	}{
		"success: basic": {
			request: ApplyTemplateRequest{Messages: []Message{{Role: "user", Content: TextContent("Hello")}}},
			prompt:  "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		"success: text_format": {
			request: ApplyTemplateRequest{
				Messages:        []Message{{Role: "user", Content: TextContent("Hello")}},
				ResponseFormat:  &ResponseFormat{Type: "text"},
				Stop:            StringValue("stop"),
				ReasoningFormat: new("none"),
				Stream:          new(true),
				Echo:            new(false),
				Logprobs:        new(true),
				TopLogprobs:     new(1),
			},
			prompt: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		"success: direct_schema": {
			request: ApplyTemplateRequest{
				Messages:       []Message{{Role: "user", Content: TextContent("Hello")}},
				ResponseFormat: &ResponseFormat{Type: "json_schema", Schema: jsontext.Value(`{"type":"object"}`)},
				Stop:           StringSlice("stop", "end"),
			},
			prompt: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		"success: options": {
			request: ApplyTemplateRequest{
				Messages:            []Message{{Role: "user", Content: PartsContent(TextContentPart("Hello"))}},
				AddGenerationPrompt: new(false),
				ChatTemplateKwargs:  map[string]jsontext.Value{"enable_thinking": jsontext.Value(`false`)},
				ParallelToolCalls:   new(false),
				ReasoningEffort:     new("none"),
			},
			prompt: "<|im_start|>user\nHello<|im_end|>\n",
		},
		"success: continue_auto": {
			request: ApplyTemplateRequest{Messages: []Message{user, assistant}, AddGenerationPrompt: new(false), ContinueFinalMessage: ContinuationAuto},
			prompt:  continued,
		},
		"success: continue_content": {
			request: ApplyTemplateRequest{Messages: []Message{user, assistant}, AddGenerationPrompt: new(false), ContinueFinalMessage: ContinuationContent},
			prompt:  continued,
		},
		"success: continue_reasoning": {
			request: ApplyTemplateRequest{
				Messages:             []Message{user, {Role: "assistant", Content: TextContent(""), ReasoningContent: new("Think of a greeting")}},
				AddGenerationPrompt:  new(false),
				ContinueFinalMessage: ContinuationReasoning,
			},
			prompt: "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n",
		},
		"success: continue_null": {
			request: ApplyTemplateRequest{
				Messages:             []Message{user},
				ContinueFinalMessage: ContinuationNull,
				ResponseFormat:       &ResponseFormat{Type: "json_object", Schema: jsontext.Value(`{"type":"object"}`)},
			},
			prompt: "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n",
		},
		"success: named_tool": {
			request: ApplyTemplateRequest{
				Messages:             []Message{weather},
				ContinueFinalMessage: ContinuationDisabled,
				Tools:                tools,
				ToolChoice:           ToolChoiceFunction("weather"),
			},
			prompt: weatherPrompt,
		},
		"success: allowed_tools": {
			request: ApplyTemplateRequest{
				Messages:   []Message{weather},
				Tools:      []Tool{deferredTool},
				ToolChoice: ToolChoiceAllowedTools("required", "weather"),
			},
			prompt: weatherPrompt,
		},
		"success: tool_mode": {
			request: ApplyTemplateRequest{
				Messages:   []Message{weather},
				Tools:      tools,
				ToolChoice: ToolChoiceMode("none"),
				ResponseFormat: &ResponseFormat{
					Type: "json_schema", JSONSchema: &JSONSchemaFormat{Schema: jsontext.Value(`{"type":"object"}`)},
				},
			},
			prompt: weatherPrompt,
		},
	}
	jsontest.CheckFallbacks[ApplyTemplateRequest](t)
	jsontest.CheckFallbacks[ApplyTemplateResponse](t)
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			fixture := "testdata/apply-template/" + name[len("success: "):]
			tokenizeCheckRequest(t, "/apply-template", fixture, tt.request)
			got := tokenizeFixture[ApplyTemplateResponse](t, fixture+".resp.json")
			if diff := gocmp.Diff(tt.prompt, got.Prompt); diff != "" {
				t.Errorf("prompt differs (-want +got):\n%s", diff)
			}
		})
	}
}

func FuzzContinuation(f *testing.F) {
	for _, stem := range []string{"continue_auto", "continue_content", "continue_reasoning", "continue_null", "named_tool"} {
		var request struct {
			ContinueFinalMessage jsontext.Value `json:"continue_final_message"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/apply-template/"+stem+".req.json"), &request); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(request.ContinueFinalMessage))
	}
	f.Fuzz(tokenizeFuzzRoundTrip[Continuation])
}

func ExampleClient_ApplyTemplate() {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		panic(err)
	}
	res, err := c.ApplyTemplate(context.Background(), &ApplyTemplateRequest{
		Messages: []Message{{Role: "user", Content: TextContent("Say hello.")}},
	})
	if err != nil {
		panic(err)
	}
	fmt.Print(res.Prompt)
}
