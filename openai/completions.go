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
	"encoding/json/jsontext"
	"iter"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
)

// CompletionRequest is the request body of POST /v1/completions.
//
// It embeds llama.SamplingParams so llama.cpp sampling and generation parameters (such as Mirostat, Seed, Stop and
// LogitBias) can be set directly in the request literal. Prompt accepts any form supported by llama.Prompt: a
// string, token ids, a mixed sequence, an object with multimodal data, or a batch of prompts.
type CompletionRequest struct {
	// Model selects the target model in router mode; single-model servers ignore it.
	Model string `json:"model,omitzero"`
	// Prompt is the prompt or batch of prompts to complete.
	Prompt llama.Prompt `json:"prompt,omitzero"`
	// MaxTokens is the OpenAI alias for SamplingParams.NPredict.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// N is the number of completions to generate per prompt (the OpenAI alias for n_cmpl).
	N *int `json:"n,omitzero"`
	// Logprobs is the number of top token probabilities to return per token (the OpenAI alias for
	// SamplingParams.NProbs).
	Logprobs *int `json:"logprobs,omitzero"`
	// Stream reports whether the server should emit server-sent events. Completions and CompletionsStream set it
	// automatically.
	Stream *bool `json:"stream,omitzero"`
	// StreamOptions configures streaming behavior.
	StreamOptions *StreamOptions `json:"stream_options,omitzero"`

	llama.SamplingParams

	Extra map[string]jsontext.Value `json:",embed"`
}

// CompletionResponse is the response of POST /v1/completions and each event of its stream.
type CompletionResponse struct {
	// ID is the unique identifier of the completion.
	ID string `json:"id"`
	// Object is "text_completion".
	Object string `json:"object"`
	// Created is the Unix timestamp in seconds when the completion was created.
	Created int64 `json:"created"`
	// Model is the model name reported by the server.
	Model string `json:"model"`
	// SystemFingerprint identifies the server build.
	SystemFingerprint string `json:"system_fingerprint"`
	// Choices are the generated completions or streamed deltas.
	Choices []CompletionChoice `json:"choices"`
	// Usage reports token counts. It is present on non-streaming responses and on the final chunk of a stream.
	Usage *Usage `json:"usage,omitzero"`
	// Timings reports prompt processing and generation speed.
	Timings *llama.Timings `json:"timings,omitzero"`
	// PromptProgress reports prompt processing progress when SamplingParams.ReturnProgress is set.
	PromptProgress *llama.PromptProgress `json:"prompt_progress,omitzero"`
	// Verbose holds the native completion response when SamplingParams.Verbose is set.
	Verbose jsontext.Value `json:"__verbose,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// CompletionChoice is one generated choice of a CompletionResponse.
type CompletionChoice struct {
	// Text is the generated text, or the next text chunk when streaming.
	Text string `json:"text"`
	// Index is the zero-based index of this choice.
	Index int `json:"index"`
	// Logprobs holds per-token probability information when Logprobs or SamplingParams.NProbs was requested.
	Logprobs *CompletionLogprobs `json:"logprobs"`
	// FinishReason is "stop" or "length" when generation has finished, or nil in intermediate stream chunks.
	FinishReason *string `json:"finish_reason"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// CompletionLogprobs holds token probability details for a CompletionChoice.
type CompletionLogprobs struct {
	// Content lists the probability report for each generated token.
	Content []llama.TokenProb `json:"content"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// Completions sends a text completion request (POST /v1/completions).
//
// Stream in req is forced to false when set to true so the response is always a single JSON object; use
// CompletionsStream for streaming.
func (c *Client) Completions(ctx context.Context, req *CompletionRequest, opts ...llama.RequestOption) (*CompletionResponse, error) {
	var body CompletionRequest
	if req != nil {
		body = *req
	}
	if body.Stream != nil && *body.Stream {
		body.Stream = new(false)
	}
	res, err := llama.Call[CompletionResponse](ctx, c.c, http.MethodPost, "/v1/completions", &body, opts...)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// CompletionsStream sends a streaming text completion request (POST /v1/completions) and yields each streamed
// chunk until the server sends "[DONE]", an error occurs, or the caller stops iterating.
//
// Stream in req is forced to true without modifying req.
func (c *Client) CompletionsStream(ctx context.Context, req *CompletionRequest, opts ...llama.RequestOption) iter.Seq2[*CompletionResponse, error] {
	var body CompletionRequest
	if req != nil {
		body = *req
	}
	body.Stream = new(true)
	return llama.Stream[CompletionResponse](ctx, c.c, http.MethodPost, "/v1/completions", &body, opts...)
}
