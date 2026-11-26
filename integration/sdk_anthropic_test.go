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

package integration

import (
	"encoding/json/jsontext"
	"net/http"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/zchee/go-llama.cpp/anthropic"
)

func TestSDKAnthropic(t *testing.T) {
	tests := map[string]struct {
		preset              *serverPreset
		stream, tool, count bool
	}{
		"success: plain":       {preset: presetAnthropic},
		"success: streamed":    {preset: presetAnthropic, stream: true},
		"success: tool use":    {preset: presetSDKTools, tool: true},
		"success: token count": {preset: presetAnthropic, count: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			path := "/v1/messages"
			if tt.count {
				path += "/count_tokens"
			}
			x := newSDKExchange(t, tt.preset.server(t), path)
			client := sdk.NewClient(option.WithoutEnvironmentDefaults(), option.WithAPIKey(testAPIKey), option.WithBaseURL(x.base), option.WithHTTPClient(&http.Client{Transport: x}), option.WithMaxRetries(0))
			text := "Tell a story."
			if tt.tool {
				text = "Call ping."
			}
			params := sdk.MessageNewParams{Model: "stories", Messages: []sdk.MessageParam{sdk.NewUserMessage(sdk.NewTextBlock(text))}, MaxTokens: 8, Temperature: sdk.Float(0)}
			local := &anthropic.MessagesRequest{Model: new("stories"), Messages: []anthropic.Message{{Role: "user", Content: anthropic.BlocksContent(anthropic.ContentBlock{Type: "text", Text: &text})}}}
			if tt.count {
				resp, err := client.Messages.CountTokens(t.Context(), sdk.MessageCountTokensParams{Model: params.Model, Messages: params.Messages})
				if err != nil {
					t.Fatalf("SDK: %v\nwire: %s", err, x.response)
				}
				x.checkResponse[anthropic.CountTokensResponse](t, resp)
				x.checkRequest(t, local)
				return
			}
			local.MaxTokens = new(8)
			local.Temperature = new(0.0)
			var opts []option.RequestOption
			if tt.tool {
				const prefill = "<tool_call>\n{\"name\":\"ping\",\"arguments\":"
				params.MaxTokens = 40
				params.Messages = append(params.Messages, sdk.NewAssistantMessage(sdk.NewTextBlock(prefill)))
				params.Tools = []sdk.ToolUnionParam{sdk.ToolUnionParamOfTool(sdk.ToolInputSchemaParam{Properties: map[string]any{"ok": map[string]any{"type": "boolean"}}, Required: []string{"ok"}, ExtraFields: map[string]any{"additionalProperties": false}}, "ping")}
				params.Tools[0].OfTool.Description = sdk.String("Check readiness")
				params.ToolChoice.OfAny = &sdk.ToolChoiceAnyParam{}
				local.MaxTokens = new(40)
				local.Messages = append(local.Messages, anthropic.Message{Role: "assistant", Content: anthropic.BlocksContent(anthropic.ContentBlock{Type: "text", Text: new(prefill)})})
				local.Tools = []anthropic.Tool{{Name: "ping", Description: new("Check readiness"), InputSchema: jsontext.Value(`{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}`)}}
				local.ToolChoice = &anthropic.ToolChoice{Type: "any"}
				local.ChatTemplateKwargs = map[string]jsontext.Value{"enable_thinking": jsontext.Value(`false`)}
				local.ExtraParams = &anthropic.ExtraParams{Echo: new(true), ParallelToolCalls: new(false)}
				opts = []option.RequestOption{option.WithJSONSet("extra_params", map[string]bool{"parallel_tool_calls": false, "echo": true}), option.WithJSONSet("chat_template_kwargs.enable_thinking", false)}
			}
			if tt.stream {
				local.Stream = new(true)
				x.checkStream[anthropic.MessageEvent](t, client.Messages.NewStreaming(t.Context(), params))
			} else {
				resp, err := client.Messages.New(t.Context(), params, opts...)
				if err != nil {
					t.Fatalf("SDK: %v\nwire: %s", err, x.response)
				}
				x.checkResponse[anthropic.MessageResponse](t, resp)
				if tt.tool {
					found := false
					for _, block := range resp.Content {
						if block.Type == "tool_use" {
							found = true
						}
					}
					if !found {
						t.Fatalf("no tool_use block: %s", x.response)
					}
				}
			}
			x.checkRequest(t, local)
		})
	}
}
