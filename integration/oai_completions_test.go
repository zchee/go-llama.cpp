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
	"github.com/zchee/go-llama.cpp/integration/harness"
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
		if len(last.Choices) == 0 {
			t.Fatalf("last chunk = %+v, want at least one choice", last)
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
		if len(res.Choices) == 0 {
			t.Fatalf("completion = %+v, want at least one choice with logprobs", res)
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

// presetOAICmplSerial runs one slot, so the prompts of a batch are processed strictly in order: a later prompt
// that fails is guaranteed to fail only after the earlier results, inside an already-open stream.
var presetOAICmplSerial = &serverPreset{
	name: "oai-completions-serial",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "stories260K"),
			Args: []string{
				"--alias", "tinyllama-2",
				"--ctx-size", "256",
				"--batch-size", "32",
				"--parallel", "1",
				"--n-predict", "64",
				"--seed", "42",
			},
		}
	},
}

func TestOAICompletionsAliasPrecedence(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	oc := openai.NewClient(c)

	// max_completion_tokens (2) must win over max_tokens (8), and n_cmpl (2) over n (1).
	res, err := oc.Completions(t.Context(), &openai.CompletionRequest{
		Model:               "tinyllama-2",
		Prompt:              llama.TextPrompt("Once upon a time"),
		MaxTokens:           new(8),
		MaxCompletionTokens: new(2),
		N:                   new(1),
		NCmpl:               new(2),
		Temperature:         new(0.0),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Choices) != 2 {
		t.Errorf("len(Choices) = %d, want 2 (n_cmpl over n)", len(res.Choices))
	}
	if res.Usage == nil || res.Usage.CompletionTokens != 4 {
		t.Errorf("Usage = %+v, want 4 completion tokens in total (max_completion_tokens over max_tokens)", res.Usage)
	}
}

func TestOAICompletionsStreamUsageNotSuppressed(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	oc := openai.NewClient(c)

	// The server ignores include_usage on this route: the final chunk carries usage either way.
	var last *openai.CompletionResponse
	for ch, err := range oc.CompletionsStream(t.Context(), &openai.CompletionRequest{
		Model:         "tinyllama-2",
		Prompt:        llama.TextPrompt("Once upon a time"),
		MaxTokens:     new(4),
		Temperature:   new(0.0),
		StreamOptions: &openai.StreamOptions{IncludeUsage: new(false)},
	}) {
		if err != nil {
			t.Fatal(err)
		}
		last = ch
	}
	if last == nil || last.Usage == nil {
		t.Fatalf("last chunk = %+v, want usage despite include_usage=false", last)
	}
	if len(last.Choices) != 1 || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "length" {
		t.Errorf("last chunk choices = %+v, want finish_reason=length", last.Choices)
	}
}

func TestOAICompletionsMultimodalRefusal(t *testing.T) {
	t.Run("error: multimodal data without a projector", func(t *testing.T) {
		c, _ := presetStories260K.server(t).Client(t)
		oc := openai.NewClient(c)
		_, err := oc.Completions(t.Context(), &openai.CompletionRequest{
			Model:     "tinyllama-2",
			Prompt:    llama.MultimodalPrompt("<__media__> describe", []byte("png")),
			MaxTokens: new(4),
		})
		apiErr, ok := errors.AsType[*llama.APIError](err)
		if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest {
			t.Fatalf("err = %v, want 400 invalid_request_error", err)
		}
		if !strings.Contains(apiErr.Message, "does not support multimodal requests") {
			t.Errorf("Message = %q, want the multimodal refusal", apiErr.Message)
		}
	})

	// With a projector loaded, the handler reads the prompt of this route as a plain string: text succeeds and
	// every other prompt form is refused, so multimodal input cannot reach the model here.
	c, _ := chatVisionPreset.server(t).Client(t)
	oc := openai.NewClient(c)

	t.Run("success: text prompt with a projector", func(t *testing.T) {
		res, err := oc.Completions(t.Context(), &openai.CompletionRequest{
			Model:       "vision",
			Prompt:      llama.TextPrompt("Hello"),
			MaxTokens:   new(2),
			Temperature: new(0.0),
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Choices) != 1 || res.Choices[0].Text == "" {
			t.Errorf("Choices = %+v, want 1 non-empty choice", res.Choices)
		}
	})

	tests := map[string]struct{ prompt llama.Prompt }{
		"error: token prompt with a projector":  {llama.TokenPrompt(1, 2, 3)},
		"error: object prompt with a projector": {llama.MultimodalPrompt("<__media__> describe", []byte("png"))},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := oc.Completions(t.Context(), &openai.CompletionRequest{Model: "vision", Prompt: tt.prompt, MaxTokens: new(2)})
			apiErr, ok := errors.AsType[*llama.APIError](err)
			if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest {
				t.Fatalf("err = %v, want 400 invalid_request_error", err)
			}
			if !strings.Contains(apiErr.Message, "type must be string") {
				t.Errorf("Message = %q, want %q", apiErr.Message, "type must be string")
			}
		})
	}
}

func TestOAICompletionsStreamErrorFrame(t *testing.T) {
	c, obs := presetOAICmplSerial.server(t).Client(t)
	oc := openai.NewClient(c)

	// A batch of a short prompt and one over the context size on a one-slot server: the short prompt streams its
	// result first, then the oversized prompt's error arrives as a frame inside the HTTP 200 stream and ends it.
	longPrompt := strings.Repeat("word ", 600)
	var chunks int
	var finished bool
	var streamErr error
	for ch, err := range oc.CompletionsStream(t.Context(), &openai.CompletionRequest{
		Model:       "tinyllama-2",
		Prompt:      llama.BatchPrompt(llama.TextPrompt("Once upon a time"), llama.TextPrompt(longPrompt)),
		MaxTokens:   new(2),
		Temperature: new(0.0),
	}) {
		if err != nil {
			streamErr = err
			break
		}
		chunks++
		if len(ch.Choices) > 0 && ch.Choices[0].FinishReason != nil {
			finished = true
		}
	}
	if chunks == 0 || !finished {
		t.Fatalf("got %d chunks (finished=%t) before the error, want the short prompt's finished result", chunks, finished)
	}
	apiErr, ok := errors.AsType[*llama.APIError](streamErr)
	if !ok {
		t.Fatalf("stream error = %v, want *llama.APIError", streamErr)
	}
	if apiErr.StatusCode != http.StatusOK || apiErr.Code != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeExceedContextSize {
		t.Errorf("APIError = %+v, want code 400 %s inside a 200 stream", apiErr, llama.ErrorTypeExceedContextSize)
	}
	if apiErr.NPromptTokens <= apiErr.NCtx || apiErr.NCtx == 0 {
		t.Errorf("context details = (%d, %d), want n_prompt_tokens > n_ctx > 0", apiErr.NPromptTokens, apiErr.NCtx)
	}
	ex := obs.Exchanges()
	if len(ex) != 1 || !ex[0].BodyClosed() {
		t.Errorf("after the error: %d exchanges, body closed=%t", len(ex), len(ex) == 1 && ex[0].BodyClosed())
	}
}
