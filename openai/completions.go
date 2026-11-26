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
	"errors"
	"iter"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/streamerr"
)

// CompletionRequest is the request body of POST /v1/completions.
//
// It embeds llama.SamplingParams so llama.cpp sampling and generation parameters (such as Mirostat, Seed, Stop and
// LogitBias) can be set directly in the request literal.
//
// Prompt accepts every form of llama.Prompt, but which forms the server takes on this route depends on how it was
// started (handle_completions_impl in tools/server/server-context.cpp and tokenize_input_subprompt in
// tools/server/server-common.cpp). Without a multimodal projector, a string, token ids, a mixed sequence and a
// batch of those are accepted, and an object prompt is accepted only without multimodal_data: media is refused
// with 400 "model does not support multimodal requests". With a projector loaded (--mmproj), the handler reads
// the prompt as a plain string, so token ids, mixed sequences, object prompts and batches are all refused with
// 400 "type must be string". Either way this route passes no media files to the model, so multimodal input never
// works here; callers that need images use ChatCompletions, whose message content parts carry them.
type CompletionRequest struct {
	// Model selects the target model in router mode; single-model servers ignore it.
	Model string `json:"model,omitzero"`
	// Prompt is the prompt or batch of prompts to complete. See the type documentation for the forms this route
	// accepts.
	Prompt llama.Prompt `json:"prompt,omitzero"`
	// MaxTokens is the OpenAI alias for SamplingParams.NPredict. The server reads the first spelling present, in
	// the order n_predict, max_completion_tokens, max_tokens.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// MaxCompletionTokens is the newer OpenAI spelling of MaxTokens. It takes precedence over MaxTokens;
	// SamplingParams.NPredict takes precedence over both.
	MaxCompletionTokens *int `json:"max_completion_tokens,omitzero"`
	// N is the number of completions to generate per prompt (the OpenAI alias for n_cmpl). NCmpl takes precedence
	// when both are present.
	N *int `json:"n,omitzero"`
	// NCmpl is the native spelling of N and takes precedence over it.
	NCmpl *int `json:"n_cmpl,omitzero"`
	// Logprobs is the number of top token probabilities to return per token (the OpenAI alias for
	// SamplingParams.NProbs). NProbs takes precedence when both are present.
	Logprobs *int `json:"logprobs,omitzero"`
	// Stream reports whether the server should emit server-sent events. Completions and CompletionsStream set it
	// automatically.
	Stream *bool `json:"stream,omitzero"`
	// StreamOptions configures streaming behavior. On this route stream_options.include_usage does not suppress
	// the usage of the last chunk: the server writes usage and finish_reason into the final chunk of each result
	// unconditionally (to_json_oaicompat in tools/server/server-task.cpp).
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
	if req == nil {
		return nil, errors.New("openai: nil completions request")
	}
	body := *req
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
// Stream in req is forced to true without modifying req. A stream that ends before the "[DONE]"
// marker yields an error matching llama.ErrIncompleteStream.
func (c *Client) CompletionsStream(ctx context.Context, req *CompletionRequest, opts ...llama.RequestOption) iter.Seq2[*CompletionResponse, error] {
	if req == nil {
		return streamerr.Once[CompletionResponse](errors.New("openai: nil completions stream request"), llama.ErrStreamConsumed)
	}
	body := *req
	body.Stream = new(true)
	return llama.Stream[CompletionResponse](ctx, c.c, http.MethodPost, "/v1/completions", &body, opts...)
}
