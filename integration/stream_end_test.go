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
	"iter"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/anthropic"
	"github.com/zchee/go-llama.cpp/openai"
)

func drainStream[T any](t *testing.T, seq iter.Seq2[*T, error]) {
	t.Helper()
	events := 0
	for _, err := range seq {
		if err != nil {
			t.Fatalf("after %d events: %v", events, err)
		}
		events++
	}
	if events == 0 {
		t.Fatal("stream yielded no events")
	}
}

// TestStreamEndToEnd runs every streaming dialect against the real server to its natural end: the
// terminal frames the client requires must arrive, so draining the stream reports no error.
func TestStreamEndToEnd(t *testing.T) {
	t.Run("native completion with siblings and batch", func(t *testing.T) {
		c, _ := presetStories260K.server(t).Client(t)
		req := &llama.CompletionRequest{
			Prompt: llama.BatchPrompt(llama.TextPrompt("Once"), llama.TextPrompt("A")),
			NCmpl:  new(2), NPredict: new(2),
		}
		drainStream(t, c.CompletionStream(t.Context(), req))
	})
	t.Run("native completion with projected fields", func(t *testing.T) {
		c, _ := presetStories260K.server(t).Client(t)
		req := &llama.CompletionRequest{Prompt: llama.TextPrompt("Once"), NPredict: new(2)}
		req.ResponseFields = []string{"content"}
		drainStream(t, c.CompletionStream(t.Context(), req))
	})
	t.Run("native infill", func(t *testing.T) {
		c, _ := infillPreset.server(t).Client(t)
		req := &llama.InfillRequest{InputPrefix: llama.InfillInput{Text: "Once"}, InputSuffix: llama.InfillInput{Text: "end"}, NPredict: new(2)}
		drainStream(t, c.InfillStream(t.Context(), req))
	})
	t.Run("openai completions", func(t *testing.T) {
		c, _ := presetStories260K.server(t).Client(t)
		req := &openai.CompletionRequest{Prompt: llama.TextPrompt("Once"), MaxTokens: new(2)}
		drainStream(t, openai.NewClient(c).CompletionsStream(t.Context(), req))
	})
	t.Run("openai chat completions", func(t *testing.T) {
		c, _ := presetStories260K.server(t).Client(t)
		drainStream(t, openai.NewClient(c).ChatCompletionsStream(t.Context(), chatRequest()))
	})
	t.Run("openai responses", func(t *testing.T) {
		c, _ := presetStories260K.server(t).Client(t)
		req := &openai.ResponsesRequest{Input: openai.ResponseInputText("Once upon a time"), MaxOutputTokens: new(4)}
		drainStream(t, openai.NewClient(c).ResponsesStream(t.Context(), req))
	})
	t.Run("anthropic messages", func(t *testing.T) {
		c, _ := presetAnthropic.server(t).Client(t, anthropicClientOptions...)
		drainStream(t, anthropic.NewClient(c).MessagesStream(t.Context(), anthropicRequest()))
	})
}
