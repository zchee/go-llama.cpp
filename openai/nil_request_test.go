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
	ctx := httptrace.WithClientTrace(t.Context(), &httptrace.ClientTrace{GetConn: func(hostPort string) {
		t.Errorf("unexpected HTTP connection attempt to %s for a nil request", hostPort)
	}})
	requestError := func(_ any, err error) error { return err }
	tests := map[string]struct {
		got  error
		want string
	}{
		"error: ChatCompletions":       {requestError(c.ChatCompletions(ctx, nil)), "openai: nil chat completions request"},
		"error: ChatCompletionsStream": {jsontest.SingleErrorStream(t, c.ChatCompletionsStream(ctx, nil), llama.ErrStreamConsumed), "openai: nil chat completions stream request"},
		"error: ChatInputTokens":       {requestError(c.ChatInputTokens(ctx, nil)), "openai: nil chat input tokens request"},
		"error: ControlChatCompletion": {requestError(c.ControlChatCompletion(ctx, nil)), "openai: nil control chat completion request"},
		"error: Completions":           {requestError(c.Completions(ctx, nil)), "openai: nil completions request"},
		"error: CompletionsStream":     {jsontest.SingleErrorStream(t, c.CompletionsStream(ctx, nil), llama.ErrStreamConsumed), "openai: nil completions stream request"},
		"error: Embeddings":            {requestError(c.Embeddings(ctx, nil)), "openai: nil embeddings request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if tt.got == nil || tt.got.Error() != tt.want {
				t.Errorf("error = %v, want %q", tt.got, tt.want)
			}
		})
	}
}
