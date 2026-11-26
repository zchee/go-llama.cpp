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
	json "encoding/json/v2"
	"net/http"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/openai"
)

func (x *sdkExchange) openAIOptions() []option.RequestOption {
	return []option.RequestOption{option.WithBaseURL(x.base + "/v1/"), option.WithAPIKey(testAPIKey), option.WithHTTPClient(&http.Client{Transport: x}), option.WithMaxRetries(0)}
}

func TestSDKChatCompletions(t *testing.T) {
	tests := map[string]struct {
		preset       *serverPreset
		stream, tool bool
	}{
		"success: plain":               {preset: presetStories260K},
		"success: streamed with usage": {preset: presetStories260K, stream: true},
		"success: tool call":           {preset: presetSDKTools, tool: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			x := newSDKExchange(t, tt.preset.server(t), "/v1/chat/completions")
			svc := sdk.NewChatCompletionService(x.openAIOptions()...)
			local := chatRequest()
			local.Model = "stories"
			params := sdk.ChatCompletionNewParams{Messages: []sdk.ChatCompletionMessageParamUnion{sdk.UserMessage("Hello!")}, Model: "stories", MaxTokens: sdk.Int(4), Temperature: sdk.Float(0)}
			var options []option.RequestOption
			if tt.tool {
				if err := json.Unmarshal(jsontest.ReadFile(t, "../openai/testdata/chat_completions/tool.req.json"), local); err != nil {
					t.Fatal(err)
				}
				params.Messages = append(params.Messages, sdk.AssistantMessage("<tool_call>\n{\"name\": \"hi\", \"arguments\":"))
				params.MaxTokens = sdk.Int(40)
				params.Tools = []sdk.ChatCompletionToolUnionParam{sdk.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{Name: "hi", Parameters: shared.FunctionParameters{"type": "object", "properties": map[string]any{}, "additionalProperties": false}})}
				params.ToolChoice.OfAuto = sdk.String("required")
				params.ParallelToolCalls = sdk.Bool(false)
				options = []option.RequestOption{option.WithJSONSet("continue_final_message", "content"), option.WithJSONSet("add_generation_prompt", false), option.WithJSONSet("echo", true), option.WithJSONSet("chat_template_kwargs.enable_thinking", false)}
			}
			if tt.stream {
				local.Stream = new(true)
				local.StreamOptions = &openai.StreamOptions{IncludeUsage: new(true)}
				params.StreamOptions.IncludeUsage = sdk.Bool(true)
				x.checkStream[openai.ChatCompletionChunk](t, svc.NewStreaming(t.Context(), params))
			} else {
				resp, err := svc.New(t.Context(), params, options...)
				if err != nil {
					t.Fatalf("SDK: %v\nwire: %s", err, x.response)
				}
				x.checkResponse[openai.ChatCompletionResponse](t, resp)
				if tt.tool && (len(resp.Choices) == 0 || len(resp.Choices[0].Message.ToolCalls) == 0) {
					t.Fatalf("no tool call: %s", x.response)
				}
			}
			x.checkRequest(t, local)
		})
	}
}

func TestSDKTextCompletion(t *testing.T) {
	x := newSDKExchange(t, presetStories260K.server(t), "/v1/completions")
	svc := sdk.NewCompletionService(x.openAIOptions()...)
	const prompt = "Once upon a time"
	params := sdk.CompletionNewParams{Model: "stories", Prompt: sdk.CompletionNewParamsPromptUnion{OfString: sdk.String(prompt)}, MaxTokens: sdk.Int(4), Temperature: sdk.Float(0)}
	resp, err := svc.New(t.Context(), params)
	if err != nil {
		t.Fatalf("SDK: %v\nwire: %s", err, x.response)
	}
	x.checkResponse[openai.CompletionResponse](t, resp)
	x.checkRequest(t, &openai.CompletionRequest{Model: "stories", Prompt: llama.TextPrompt(prompt), MaxTokens: new(4), Temperature: new(0.0)})
}

func TestSDKEmbeddings(t *testing.T) {
	tests := map[string]struct{ format llama.EmbeddingEncodingFormat }{"success: float": {llama.EmbeddingEncodingFloat}, "success: base64": {llama.EmbeddingEncodingBase64}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			x := newSDKExchange(t, presetEmbeddingPooled.server(t), "/v1/embeddings")
			svc := sdk.NewEmbeddingService(x.openAIOptions()...)
			const model = "bert-bge-small"
			params := sdk.EmbeddingNewParams{Model: model, Input: sdk.EmbeddingNewParamsInputUnion{OfString: sdk.String("hello")}, EncodingFormat: sdk.EmbeddingNewParamsEncodingFormat(tt.format)}
			resp, err := svc.New(t.Context(), params)
			if err != nil {
				t.Fatalf("SDK: %v\nwire: %s", err, x.response)
			}
			local := jsontest.RoundTrip[openai.EmbeddingResponse](t, x.response)
			if local.Object != string(resp.Object) || local.Model != resp.Model || int64(local.Usage.PromptTokens) != resp.Usage.PromptTokens || int64(local.Usage.TotalTokens) != resp.Usage.TotalTokens || len(local.Data) != len(resp.Data) || len(local.Data) == 0 {
				t.Fatalf("embedding envelope mismatch: SDK %+v; wire: %s", resp, x.response)
			}
			for i, v := range local.Data {
				sdkItem := resp.Data[i]
				if v.Embedding.Format() != tt.format || v.Object != string(sdkItem.Object) || int64(v.Index) != sdkItem.Index || v.Embedding.Len() == 0 {
					t.Fatalf("embedding %d metadata mismatch: SDK %+v; wire: %s", i, sdkItem, x.response)
				}
				if diff := gocmp.Diff(sdkItem.Embedding, v.Embedding.Values()); diff != "" {
					t.Errorf("embedding %d (-SDK +library):\n%s\nwire: %s", i, diff, x.response)
				}
			}
			x.checkRequest(t, &openai.EmbeddingRequest{Model: new(model), Input: llama.EmbeddingText("hello"), EncodingFormat: &tt.format, Normalize: new(2)}, "embd_normalize")
		})
	}
}

func TestSDKResponses(t *testing.T) {
	tests := map[string]struct{ stream bool }{"success: plain": {}, "success: streamed": {true}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			x := newSDKExchange(t, presetStories260K.server(t), "/v1/responses")
			svc := responses.NewResponseService(x.openAIOptions()...)
			const prompt = "Once upon a time"
			params := responses.ResponseNewParams{Model: "stories", Input: responses.ResponseNewParamsInputUnion{OfString: sdk.String(prompt)}, MaxOutputTokens: sdk.Int(4), Temperature: sdk.Float(0)}
			local := &openai.ResponsesRequest{Model: "stories", Input: openai.ResponseInputText(prompt), MaxOutputTokens: new(4), Temperature: new(0.0)}
			if tt.stream {
				local.Stream = new(true)
				x.checkStream[openai.ResponseEvent](t, svc.NewStreaming(t.Context(), params))
			} else {
				resp, err := svc.New(t.Context(), params)
				if err != nil {
					t.Fatalf("SDK: %v\nwire: %s", err, x.response)
				}
				x.checkResponse[openai.Response](t, resp)
			}
			x.checkRequest(t, local)
		})
	}
}
