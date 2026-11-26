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
	"errors"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

func TestOAICompletions(t *testing.T) {
	srv := presetStories260K.server(t)

	t.Run("success: non-stream forces stream false without mutating request", func(t *testing.T) {
		c, obs := srv.Client(t)
		oc := openai.NewClient(c)
		req := &openai.CompletionRequest{
			Model:       "tinyllama-2",
			Prompt:      llama.TextPrompt("Once upon a time"),
			MaxTokens:   new(8),
			Stream:      new(true),
			Temperature: new(0.0),
			Seed:        new(int64(42)),
		}
		res, err := oc.Completions(t.Context(), req)
		if err != nil {
			t.Fatal(err)
		}
		if req.Stream == nil || !*req.Stream {
			t.Errorf("Completions mutated caller's req.Stream to %v", req.Stream)
		}
		if res.Object != "text_completion" || len(res.Choices) != 1 || res.Choices[0].Text == "" {
			t.Fatalf("unexpected response: %+v", res)
		}
		if res.Timings == nil || res.Usage == nil {
			t.Errorf("missing Timings (%v) or Usage (%v)", res.Timings, res.Usage)
		}
		ex := obs.Exchanges()
		if len(ex) != 1 || !ex[0].BodyClosed() {
			t.Errorf("exchanges = %+v, want 1 closed exchange", ex)
		}
	})

	t.Run("success: streaming completes with [DONE], closes body, and rejects second range", func(t *testing.T) {
		c, obs := srv.Client(t)
		oc := openai.NewClient(c)
		req := &openai.CompletionRequest{
			Model:          "tinyllama-2",
			Prompt:         llama.TextPrompt("Once upon a time"),
			MaxTokens:      new(8),
			Temperature:    new(0.0),
			ReturnProgress: new(true),
		}
		seq := oc.CompletionsStream(t.Context(), req)
		if req.Stream != nil {
			t.Errorf("CompletionsStream mutated caller's req.Stream to %v", req.Stream)
		}
		if len(obs.Exchanges()) != 0 {
			t.Fatalf("CompletionsStream sent a request before ranging")
		}

		var text strings.Builder
		var last *openai.CompletionResponse
		count := 0
		for ch, err := range seq {
			if err != nil {
				t.Fatalf("stream error after %d chunks: %v", count, err)
			}
			count++
			last = ch
			if len(ch.Choices) > 0 {
				text.WriteString(ch.Choices[0].Text)
			}
		}
		if count < 2 || text.Len() == 0 {
			t.Fatalf("got %d chunks with text %q, want >= 2 chunks and non-empty text", count, text.String())
		}
		if last.Usage == nil || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "length" {
			t.Errorf("last chunk = %+v, want usage and finish_reason=length", last)
		}
		ex := obs.Exchanges()
		if len(ex) != 1 || !ex[0].BodyClosed() {
			t.Fatalf("after stream: %d exchanges, body closed=%t", len(ex), len(ex) == 1 && ex[0].BodyClosed())
		}

		var errs []error
		for ch, err := range seq {
			if ch != nil {
				t.Errorf("second range yielded chunk %+v", ch)
			}
			errs = append(errs, err)
		}
		if len(errs) != 1 || !errors.Is(errs[0], llama.ErrStreamConsumed) {
			t.Errorf("second range errors = %v, want [ErrStreamConsumed]", errs)
		}
		if len(obs.Exchanges()) != 1 {
			t.Errorf("second range sent a request")
		}
	})

	t.Run("success: streaming early break closes body", func(t *testing.T) {
		c, obs := srv.Client(t)
		oc := openai.NewClient(c)
		n := 0
		for _, err := range oc.CompletionsStream(t.Context(), &openai.CompletionRequest{
			Model:     "tinyllama-2",
			Prompt:    llama.TextPrompt("Once upon a time"),
			MaxTokens: new(16),
			IgnoreEOS: new(true),
		}) {
			if err != nil {
				t.Fatal(err)
			}
			n++
			if n == 2 {
				break
			}
		}
		ex := obs.Exchanges()
		if len(ex) != 1 || !ex[0].BodyClosed() {
			t.Fatalf("after break: %d exchanges, body closed=%t", len(ex), len(ex) == 1 && ex[0].BodyClosed())
		}
	})

	t.Run("success: logprobs and post_sampling_probs", func(t *testing.T) {
		c, _ := srv.Client(t)
		oc := openai.NewClient(c)
		res, err := oc.Completions(t.Context(), &openai.CompletionRequest{
			Model:             "tinyllama-2",
			Prompt:            llama.TextPrompt("Once upon a time"),
			MaxTokens:         new(4),
			Logprobs:          new(2),
			Temperature:       new(0.0),
			PostSamplingProbs: new(true),
		})
		if err != nil {
			t.Fatal(err)
		}
		lp := res.Choices[0].Logprobs
		if lp == nil || len(lp.Content) == 0 || lp.Content[0].Prob == nil || len(lp.Content[0].TopProbs) == 0 {
			t.Errorf("Logprobs = %+v, want post-sampling probabilities", lp)
		}
	})

	t.Run("success: multiple choices and prompt forms", func(t *testing.T) {
		c, _ := srv.Client(t)
		oc := openai.NewClient(c)

		tests := map[string]struct {
			req         *openai.CompletionRequest
			wantChoices int
		}{
			"success: n=2": {
				req: &openai.CompletionRequest{
					Model:       "tinyllama-2",
					Prompt:      llama.TextPrompt("Once upon a time"),
					MaxTokens:   new(4),
					N:           new(2),
					Temperature: new(0.0),
				},
				wantChoices: 2,
			},
			"success: batch prompt": {
				req: &openai.CompletionRequest{
					Model:       "tinyllama-2",
					Prompt:      llama.BatchPrompt(llama.TextPrompt("Once upon"), llama.TextPrompt("Hello")),
					MaxTokens:   new(4),
					Temperature: new(0.0),
				},
				wantChoices: 2,
			},
			"success: token prompt": {
				req: &openai.CompletionRequest{
					Model:       "tinyllama-2",
					Prompt:      llama.TokenPrompt(1, 2, 3),
					MaxTokens:   new(4),
					Temperature: new(0.0),
				},
				wantChoices: 1,
			},
			"success: mixed prompt": {
				req: &openai.CompletionRequest{
					Model:       "tinyllama-2",
					Prompt:      llama.MixedPrompt(llama.TokenPart(1), llama.TextPart("Once upon")),
					MaxTokens:   new(4),
					Temperature: new(0.0),
				},
				wantChoices: 1,
			},
		}

		for name, tt := range tests {
			t.Run(name, func(t *testing.T) {
				res, err := oc.Completions(t.Context(), tt.req)
				if err != nil {
					t.Fatal(err)
				}
				if len(res.Choices) != tt.wantChoices {
					t.Errorf("len(Choices) = %d, want %d", len(res.Choices), tt.wantChoices)
				}
			})
		}
	})

	t.Run("error: missing prompt and exceed context size", func(t *testing.T) {
		c, _ := srv.Client(t)
		oc := openai.NewClient(c)

		_, err := oc.Completions(t.Context(), &openai.CompletionRequest{Model: "tinyllama-2"})
		apiErr, ok := errors.AsType[*llama.APIError](err)
		if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest {
			t.Errorf("missing prompt err = %v, want 400 invalid_request_error", err)
		}

		longPrompt := strings.Repeat("word ", 600)
		_, err = oc.Completions(t.Context(), &openai.CompletionRequest{
			Model:     "tinyllama-2",
			Prompt:    llama.TextPrompt(longPrompt),
			MaxTokens: new(4),
		})
		apiErr, ok = errors.AsType[*llama.APIError](err)
		if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeExceedContextSize {
			t.Errorf("long prompt err = %v, want 400 exceed_context_size_error", err)
		}
		if apiErr != nil && (apiErr.NCtx != 256 || apiErr.NPromptTokens <= apiErr.NCtx) {
			t.Errorf("exceed_context_size details = (%d, %d), want (>256, 256)", apiErr.NPromptTokens, apiErr.NCtx)
		}

		var streamErr error
		for _, err := range oc.CompletionsStream(t.Context(), &openai.CompletionRequest{
			Model:  "tinyllama-2",
			Prompt: llama.TextPrompt(longPrompt),
		}) {
			streamErr = err
			break
		}
		streamAPIErr, ok := errors.AsType[*llama.APIError](streamErr)
		if !ok || streamAPIErr.StatusCode != http.StatusBadRequest || streamAPIErr.Type != llama.ErrorTypeExceedContextSize {
			t.Errorf("stream long prompt err = %v, want 400 exceed_context_size_error", streamErr)
		}
	})
}
