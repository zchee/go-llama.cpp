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
	"context"
	"net/http/httptrace"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestNilRequiredRequest(t *testing.T) {
	base, err := llama.New("http://127.0.0.1:1")
	if err != nil {
		t.Fatal(err)
	}
	c := NewClient(base)
	ctx := func(t *testing.T) context.Context {
		return httptrace.WithClientTrace(t.Context(), &httptrace.ClientTrace{GetConn: func(hostPort string) {
			t.Errorf("unexpected HTTP connection attempt to %s for a nil request", hostPort)
		}})
	}
	requestError := func(_ any, err error) error { return err }
	tests := map[string]struct {
		call func(*testing.T) error
		want string
	}{
		"error: ChatCompletions": {func(t *testing.T) error { return requestError(c.ChatCompletions(ctx(t), nil)) }, "openai: nil chat completions request"},
		"error: ChatCompletionsStream": {func(t *testing.T) error {
			return jsontest.SingleErrorStream(t, c.ChatCompletionsStream(ctx(t), nil), llama.ErrStreamConsumed)
		}, "openai: nil chat completions stream request"},
		"error: ChatInputTokens":       {func(t *testing.T) error { return requestError(c.ChatInputTokens(ctx(t), nil)) }, "openai: nil chat input tokens request"},
		"error: ControlChatCompletion": {func(t *testing.T) error { return requestError(c.ControlChatCompletion(ctx(t), nil)) }, "openai: nil control chat completion request"},
		"error: Completions":           {func(t *testing.T) error { return requestError(c.Completions(ctx(t), nil)) }, "openai: nil completions request"},
		"error: CompletionsStream": {func(t *testing.T) error {
			return jsontest.SingleErrorStream(t, c.CompletionsStream(ctx(t), nil), llama.ErrStreamConsumed)
		}, "openai: nil completions stream request"},
		"error: Embeddings":           {func(t *testing.T) error { return requestError(c.Embeddings(ctx(t), nil)) }, "openai: nil embeddings request"},
		"error: Responses":            {func(t *testing.T) error { return requestError(c.Responses(ctx(t), nil)) }, "openai: nil responses request"},
		"error: ResponsesInputTokens": {func(t *testing.T) error { return requestError(c.ResponsesInputTokens(ctx(t), nil)) }, "openai: nil responses input tokens request"},
		"error: ResponsesStream": {func(t *testing.T) error {
			return jsontest.SingleErrorStream(t, c.ResponsesStream(ctx(t), nil), llama.ErrStreamConsumed)
		}, "openai: nil responses stream request"},
		"error: Speech": {func(t *testing.T) error { return requestError(c.Speech(ctx(t), nil)) }, "openai: nil speech request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := tt.call(t); got == nil || got.Error() != tt.want {
				t.Errorf("error = %v, want %q", got, tt.want)
			}
		})
	}
}
