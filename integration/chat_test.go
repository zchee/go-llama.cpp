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
	json "encoding/json/v2"
	"errors"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/openai"
)

func chatRequest() *openai.ChatCompletionRequest {
	return &openai.ChatCompletionRequest{Messages: []openai.Message{{Role: "user", Content: openai.TextContent("Hello!")}}, Temperature: new(0.0), MaxTokens: new(4)}
}

var chatControlPreset = &serverPreset{name: "chat-control", options: func(t testing.TB) harness.Options {
	return harness.Options{Model: harness.Model(t, "stories260K"), Args: []string{"--alias", "stories", "--ctx-size", "2048", "--parallel", "1", "--threads-http", "4", "--threads", "1", "--jinja", "--chat-template", chatThinkingTemplate}}
}}

var chatVisionPreset = &serverPreset{name: "chat-vision", options: func(t testing.TB) harness.Options {
	return harness.Options{Model: harness.Model(t, "tinygemma3"), Args: []string{"--alias", "vision", "--ctx-size", "2048", "--parallel", "1", "--mmproj", harness.Model(t, "tinygemma3-mmproj")}}
}}

// chatThinkingTemplate renders reasoning_content inside <think> tags and opens a reasoning block in the
// generation prompt. The server detects the tags and only then reads reasoning_control from a request.
const chatThinkingTemplate = `{%- for m in messages -%}
<|im_start|>{{ m.role }}
{% if m.reasoning_content %}<think>
{{ m.reasoning_content }}
</think>
{% endif %}{{ m.content }}<|im_end|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
<think>
{%- endif -%}`

func TestChatReasoningControlArmed(t *testing.T) {
	c, _ := chatControlPreset.server(t).Client(t)
	chat := openai.NewClient(c)
	req := chatRequest()
	req.MaxTokens = new(512)
	req.IgnoreEOS = new(true)
	req.ReasoningControl = new(true)
	for chunk, err := range chat.ChatCompletionsStream(t.Context(), req) {
		if err != nil {
			t.Fatal(err)
		}
		if chunk.ID == "" {
			t.Fatal("initial chunk has no ID")
		}
		ctl, err := chat.ControlChatCompletion(t.Context(), &openai.ControlChatCompletionRequest{ID: chunk.ID, Action: "reasoning_end"})
		if err != nil {
			t.Fatal(err)
		}
		if !ctl.Success {
			t.Fatalf("armed live stream control: %+v", ctl)
		}
		return
	}
	t.Fatal("stream ended before the first chunk")
}

func TestChatIntegerLogprobs(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	chat := openai.NewClient(c)
	req := chatRequest()
	req.NProbs = new(2)
	res, err := chat.ChatCompletions(t.Context(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Choices) == 0 || res.Choices[0].Logprobs == nil || len(res.Choices[0].Logprobs.Content) == 0 {
		t.Fatalf("n_probs returned no probabilities: %+v", res.Choices)
	}
	for _, prob := range res.Choices[0].Logprobs.Content {
		if len(prob.TopLogprobs) != *req.NProbs {
			t.Fatalf("token %q has %d alternatives, want %d", prob.Token, len(prob.TopLogprobs), *req.NProbs)
		}
	}
}

func TestChatStreamControl(t *testing.T) {
	c, obs := chatControlPreset.server(t).Client(t)
	chat := openai.NewClient(c)
	req := chatRequest()
	req.MaxTokens = new(512)
	req.IgnoreEOS = new(true)
	req.Stream = new(false)
	req.StreamOptions = &openai.StreamOptions{IncludeUsage: new(true)}
	var id string
	var usage *llama.Usage
	var finished bool
	for chunk, err := range chat.ChatCompletionsStream(t.Context(), req, llama.WithQuery("client", "chat")) {
		if err != nil {
			t.Fatal(err)
		}
		if id == "" {
			id = chunk.ID
			if id == "" {
				t.Fatal("initial chunk has no ID")
			}
			ctl, err := chat.ControlChatCompletion(t.Context(), &openai.ControlChatCompletionRequest{ID: id, Action: "reasoning_end"})
			if err != nil {
				t.Fatal(err)
			}
			if ctl.Success || ctl.Message == nil || *ctl.Message != "reasoning control not enabled for this completion" {
				t.Fatalf("unarmed live stream control = %+v", ctl)
			}
		} else if chunk.ID != id {
			t.Errorf("chunk ID changed from %q to %q", id, chunk.ID)
		}
		for i := range chunk.Choices {
			if chunk.Choices[i].FinishReason != nil {
				finished = true
			}
		}
		if chunk.Usage != nil {
			if !finished || len(chunk.Choices) != 0 {
				t.Errorf("usage precedes finish or contains choices: %+v", chunk)
			}
			usage = chunk.Usage
		}
	}
	if !finished || usage == nil || usage.CompletionTokens != *req.MaxTokens {
		t.Fatalf("finished=%t, usage=%+v", finished, usage)
	}
	if *req.Stream {
		t.Error("streaming changed request.Stream")
	}
	ctl, err := chat.ControlChatCompletion(t.Context(), &openai.ControlChatCompletionRequest{ID: id, Action: "reasoning_end"})
	if err != nil {
		t.Fatal(err)
	}
	if ctl.Success || ctl.Message == nil || *ctl.Message != "no active completion for this id" {
		t.Fatalf("finished stream control = %+v", ctl)
	}
	exchanges := obs.Exchanges()
	if len(exchanges) != 3 || exchanges[0].Query.Get("client") != "chat" || !exchanges[0].BodyClosed() {
		t.Errorf("unexpected exchanges: %+v", exchanges)
	}
}

func TestChatCompletionAndCount(t *testing.T) {
	tests := map[string]struct {
		server *serverPreset
		image  bool
	}{
		"success: text prompt":     {server: presetStories260K},
		"success: projected image": {server: chatVisionPreset, image: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := tt.server.server(t).Client(t)
			chat := openai.NewClient(c)
			req := chatRequest()
			if tt.image {
				if err := json.Unmarshal(jsontest.ReadFile(t, "../openai/testdata/chat_completions/image.req.json"), req); err != nil {
					t.Fatal(err)
				}
			}
			req.Stream = new(true)
			res, err := chat.ChatCompletions(t.Context(), req)
			if err != nil {
				t.Fatal(err)
			}
			if !*req.Stream {
				t.Error("completion changed request.Stream")
			}
			count, err := chat.ChatInputTokens(t.Context(), req)
			if err != nil {
				t.Fatal(err)
			}
			if count.InputTokens != int64(res.Usage.PromptTokens) {
				t.Errorf("input_tokens = %d, completion usage reports %d prompt tokens", count.InputTokens, res.Usage.PromptTokens)
			}
		})
	}
}

func TestChatErrors(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	chat := openai.NewClient(c)
	tests := map[string]struct {
		req  *openai.ChatCompletionRequest
		want llama.ErrorType
	}{
		"error: top logprobs without logprobs": {req: &openai.ChatCompletionRequest{Messages: chatRequest().Messages, TopLogprobs: new(2)}, want: llama.ErrorTypeInvalidRequest},
		"error: oversized conversation":        {req: &openai.ChatCompletionRequest{Messages: []openai.Message{{Role: "user", Content: openai.TextContent(strings.Repeat("hello ", 600))}}}, want: llama.ErrorTypeExceedContextSize},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := chat.ChatCompletions(t.Context(), tt.req)
			api, ok := errors.AsType[*llama.APIError](err)
			if !ok || api.Type != tt.want || api.StatusCode != http.StatusBadRequest {
				t.Fatalf("error=%v, want HTTP 400 %s", err, tt.want)
			}
			var n int
			for chunk, err := range chat.ChatCompletionsStream(t.Context(), tt.req) {
				n++
				api, ok := errors.AsType[*llama.APIError](err)
				if !ok || api.Type != tt.want || chunk != nil {
					t.Fatalf("chunk=%+v, err=%v", chunk, err)
				}
			}
			if n != 1 {
				t.Errorf("stream yielded %d results on failure", n)
			}
		})
	}
}

func TestChatAliases(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	tests := map[string]struct{ path, object string }{
		"success: unversioned completion":   {path: "/chat/completions", object: "chat.completion"},
		"success: unversioned input tokens": {path: "/chat/completions/input_tokens", object: "response.input_tokens"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			res, err := llama.Call[map[string]jsontext.Value](t.Context(), c, http.MethodPost, tt.path, chatRequest())
			if err != nil {
				t.Fatal(err)
			}
			var object string
			if err := json.Unmarshal(res["object"], &object); err != nil {
				t.Fatal(err)
			}
			if object != tt.object {
				t.Errorf("object = %q, want %q", object, tt.object)
			}
		})
	}
}
