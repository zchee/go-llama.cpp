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
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func chatWire[T any](t *testing.T, data []byte) *T {
	t.Helper()
	jsontest.CheckFallbacks[T](t)
	jsontest.NoUnknown[T](t, data, nil)
	jsontest.InjectUnknown[T](t, data)
	return jsontest.RoundTrip[T](t, data)
}

func chatRequestWire[T any](t *testing.T, c *llama.Client, path string, data []byte) {
	t.Helper()
	body := chatWire[T](t, data)
	req, err := c.NewRequest(t.Context(), http.MethodPost, path, body, llama.WithModelQuery("stories"))
	if err != nil {
		t.Fatal(err)
	}
	defer req.Body.Close()
	if req.Method != http.MethodPost || req.URL.Path != path || req.URL.RawQuery != "model=stories" {
		t.Fatalf("request = %s %s", req.Method, req.URL)
	}
	wire, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatal(err)
	}
	want, got := jsontext.Value(data).Clone(), jsontext.Value(wire)
	if err := want.Canonicalize(); err != nil {
		t.Fatal(err)
	}
	if err := got.Canonicalize(); err != nil {
		t.Fatal(err)
	}
	if diff := gocmp.Diff(string(want), string(got)); diff != "" {
		t.Errorf("request body (-recorded +sent):\n%s", diff)
	}
}

func TestChatCompletionWire(t *testing.T) {
	c, err := llama.New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		fixture   string
		stream    bool
		errorType llama.ErrorType
	}{
		"success: plain text":                                   {fixture: "plain"},
		"success: streamed probabilities and usage":             {fixture: "stream", stream: true},
		"success: log probabilities":                            {fixture: "logprobs"},
		"success: post-sampling probabilities and debug fields": {fixture: "post_sampling"},
		"success: multiple choices":                             {fixture: "multiple"},
		"success: tool call":                                    {fixture: "tool"},
		"success: streamed tool call":                           {fixture: "tool_stream", stream: true},
		"success: named tool":                                   {fixture: "tool_named"},
		"success: JSON object":                                  {fixture: "json_object"},
		"success: object with schema":                           {fixture: "object_schema"},
		"success: direct JSON schema":                           {fixture: "direct_schema"},
		"success: wrapped JSON schema":                          {fixture: "schema"},
		"success: reasoning":                                    {fixture: "reasoning"},
		"success: streamed reasoning":                           {fixture: "reasoning_stream", stream: true},
		"success: image parts":                                  {fixture: "image"},
		"success: continue assistant content":                   {fixture: "continuation"},
		"success: continue automatically":                       {fixture: "auto"},
		"error: logprobs not enabled":                           {fixture: "request_error", errorType: llama.ErrorTypeInvalidRequest},
		"error: oversized conversation":                         {fixture: "context_error", errorType: llama.ErrorTypeExceedContextSize},
		"error: rejected before streaming":                      {fixture: "stream_error", errorType: llama.ErrorTypeExceedContextSize},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/chat_completions/" + tt.fixture
			chatRequestWire[ChatCompletionRequest](t, c, chatCompletionsPath, jsontest.ReadFile(t, prefix+".req.json"))
			if tt.stream {
				chatStreamWire(t, prefix, tt.fixture)
				return
			}
			data := jsontest.ReadFile(t, prefix+".resp.json")
			if tt.errorType != "" {
				chatErrorWire(t, data, tt.errorType)
				return
			}
			res := chatWire[ChatCompletionResponse](t, data)
			if res.Object != "chat.completion" || len(res.Choices) == 0 || res.Timings == nil || res.Usage.CompletionTokens <= 0 {
				t.Fatalf("invalid completion: %+v", res)
			}
			choice := res.Choices[0]
			switch tt.fixture {
			case "schema":
				if choice.Message.Content.Text() != `"ok"` {
					t.Errorf("schema-constrained JSON = %q", choice.Message.Content.Text())
				}
			case "tool", "tool_named":
				if len(choice.Message.ToolCalls) != 1 || choice.Message.ToolCalls[0].Function == nil || choice.Message.ToolCalls[0].Function.Arguments == nil || !jsontext.Value(*choice.Message.ToolCalls[0].Function.Arguments).IsValid() {
					t.Fatalf("invalid tool call: %+v", choice.Message)
				}
			case "reasoning":
				if choice.Message.ReasoningContent == nil || *choice.Message.ReasoningContent == "" {
					t.Error("missing assistant reasoning")
				}
			case "multiple":
				if len(res.Choices) != 2 || res.Choices[1].Index != 1 {
					t.Errorf("choices = %+v", res.Choices)
				}
			}
		})
	}
}

func chatErrorWire(t *testing.T, data []byte, want llama.ErrorType) {
	t.Helper()
	e := llama.ReadAPIError(&http.Response{StatusCode: http.StatusBadRequest, Body: io.NopCloser(bytes.NewReader(data))})
	if e.Type != want || e.Code != http.StatusBadRequest || e.Message == "" {
		t.Errorf("error = %+v, want %s", e, want)
	}
	if want == llama.ErrorTypeExceedContextSize && (e.NCtx <= 0 || e.NPromptTokens <= e.NCtx) {
		t.Errorf("missing context limits: %+v", e)
	}
}

func chatStreamWire(t *testing.T, prefix, fixture string) {
	t.Helper()
	data := jsontest.ReadFile(t, prefix+".resp.sse")
	if !bytes.HasSuffix(bytes.TrimSpace(data), []byte("data: [DONE]")) {
		t.Fatal("missing stream terminator")
	}
	events := jsontest.EventData(t, data)
	var n int
	var final *ChatCompletionChunk
	var args, reasoning strings.Builder
	var finished bool
	for chunk, err := range llama.DecodeStream[ChatCompletionChunk](bytes.NewReader(data), 0) {
		if err != nil {
			t.Fatal(err)
		}
		if n >= len(events) {
			t.Fatal("decoder yielded an extra event")
		}
		want := chatWire[ChatCompletionChunk](t, events[n].Data)
		if diff := gocmp.Diff(want, chunk); diff != "" {
			t.Errorf("chunk %d (-want +got):\n%s", n, diff)
		}
		n++
		final = chunk
		for i := range chunk.Choices {
			choice := &chunk.Choices[i]
			if choice.FinishReason != nil {
				finished = true
			}
			if choice.Delta.ReasoningContent != nil {
				reasoning.WriteString(*choice.Delta.ReasoningContent)
			}
			for _, call := range choice.Delta.ToolCalls {
				if call.Index == nil || *call.Index != 0 {
					t.Fatalf("tool index = %v", call.Index)
				}
				if call.Function != nil && call.Function.Arguments != nil {
					args.WriteString(*call.Function.Arguments)
				}
			}
		}
	}
	if n != len(events) || final == nil || !finished {
		t.Fatalf("decoded %d/%d events, finished=%t", n, len(events), finished)
	}
	switch fixture {
	case "stream":
		if len(final.Choices) != 0 || final.Usage == nil || final.Timings == nil {
			t.Errorf("invalid usage chunk: %+v", final)
		}
	case "tool_stream":
		if !jsontext.Value(args.String()).IsValid() {
			t.Errorf("invalid accumulated arguments: %q", args.String())
		}
	case "reasoning_stream":
		if reasoning.Len() == 0 {
			t.Error("missing reasoning delta")
		}
	}
}

func TestChatCountAndControlWire(t *testing.T) {
	c, err := llama.New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		dir, name, path  string
		success, invalid bool
	}{
		"success: count":                 {dir: "chat_completions_input_tokens", name: "basic", path: "/input_tokens"},
		"error: invalid conversation":    {dir: "chat_completions_input_tokens", name: "invalid_chat", path: "/input_tokens", invalid: true},
		"success: enabled live control":  {dir: "chat_completions_control", name: "armed", path: "/control", success: true},
		"success: disabled live control": {dir: "chat_completions_control", name: "unarmed", path: "/control"},
		"success: inactive completion":   {dir: "chat_completions_control", name: "finished", path: "/control"},
		"error: unsupported action":      {dir: "chat_completions_control", name: "bad_action", path: "/control", invalid: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/" + tt.dir + "/" + tt.name
			body := jsontest.ReadFile(t, prefix+".req.json")
			data := jsontest.ReadFile(t, prefix+".resp.json")
			if tt.path == "/control" {
				chatRequestWire[ControlChatCompletionRequest](t, c, chatCompletionsPath+tt.path, body)
			} else {
				chatRequestWire[ChatCompletionRequest](t, c, chatCompletionsPath+tt.path, body)
			}
			if tt.invalid {
				chatErrorWire(t, data, llama.ErrorTypeInvalidRequest)
				return
			}
			if tt.path == "/input_tokens" {
				res := chatWire[InputTokensResponse](t, data)
				if res.InputTokens <= 0 || res.Object != "response.input_tokens" {
					t.Errorf("count = %+v", res)
				}
				return
			}
			res := chatWire[ControlChatCompletionResponse](t, data)
			if res.Success != tt.success || (res.Message == nil) != tt.success {
				t.Errorf("control = %+v, want success %t", res, tt.success)
			}
		})
	}
}

func FuzzChatContinuation(f *testing.F) {
	for _, name := range []string{"plain", "auto", "logprobs", "continuation", "reasoning"} {
		var req map[string]jsontext.Value
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/chat_completions/"+name+".req.json"), &req); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(req["continue_final_message"]))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var a, b ChatContinuation
		if err := json.Unmarshal(data, &a); err != nil {
			return
		}
		encoded, err := json.Marshal(a)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(encoded, &b); err != nil {
			t.Fatal(err)
		}
		if a != b {
			t.Fatalf("continuation changed from %v to %v", a, b)
		}
	})
}

func ExampleClient_ChatCompletions() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	chat := NewClient(c)
	req := &ChatCompletionRequest{Messages: []Message{{Role: "user", Content: TextContent("Hello!")}}, MaxTokens: new(32)}
	res, err := chat.ChatCompletions(context.Background(), req)
	if err != nil {
		panic(err)
	}
	for i := range res.Choices {
		fmt.Println(res.Choices[i].Message.Content.Text())
	}
}
