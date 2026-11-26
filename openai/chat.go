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

const chatCompletionsPath = "/v1/chat/completions"

// ChatCompletionRequest is a chat conversation and its generation options. ChatInputTokens accepts the same body.
type ChatCompletionRequest struct {
	llama.SamplingParams
	// Model selects the model in router mode.
	Model string `json:"model,omitzero"`
	// Messages is the conversation in chronological order.
	Messages []Message `json:"messages"`
	// MaxTokens limits the number of generated tokens.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// MaxCompletionTokens is another spelling of the generation limit.
	MaxCompletionTokens *int `json:"max_completion_tokens,omitzero"`
	// N is the number of choices to generate.
	N *int `json:"n,omitzero"`
	// NCmpl is the native spelling of N.
	NCmpl *int `json:"n_cmpl,omitzero"`
	// Stream requests SSE on the wire. The method used determines the response mode.
	Stream *bool `json:"stream,omitzero"`
	// StreamOptions controls whether the stream includes a final usage chunk.
	StreamOptions *StreamOptions `json:"stream_options,omitzero"`
	// Tools describes the functions available to the model.
	Tools []Tool `json:"tools,omitzero"`
	// ToolChoice selects how the model may call the supplied tools.
	ToolChoice ToolChoice `json:"tool_choice,omitzero"`
	// ParallelToolCalls allows multiple calls when supported by the template.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitzero"`
	// ResponseFormat constrains the output to text, a JSON object, or a JSON schema.
	ResponseFormat *ResponseFormat `json:"response_format,omitzero"`
	// Logprobs requests token probabilities in each choice.
	Logprobs *bool `json:"logprobs,omitzero"`
	// TopLogprobs is the number of alternatives per token when Logprobs is true.
	TopLogprobs *int `json:"top_logprobs,omitzero"`
	// ChatTemplateKwargs overrides named Jinja template arguments, preserving their JSON types.
	ChatTemplateKwargs map[string]jsontext.Value `json:"chat_template_kwargs,omitzero"`
	// ReasoningEffort is passed to the template; "none" disables reasoning.
	ReasoningEffort *string `json:"reasoning_effort,omitzero"`
	// ThinkingBudgetTokens is the alternative spelling of reasoning_budget_tokens.
	ThinkingBudgetTokens *int `json:"thinking_budget_tokens,omitzero"`
	// ReasoningBudgetEndTag is the singular alternative to reasoning_budget_end_tags.
	ReasoningBudgetEndTag *string `json:"reasoning_budget_end_tag,omitzero"`
	// AddGenerationPrompt appends the assistant prefix before generation.
	AddGenerationPrompt *bool `json:"add_generation_prompt,omitzero"`
	// ContinueFinalMessage continues the final assistant message instead of starting another.
	ContinueFinalMessage ChatContinuation `json:"continue_final_message,omitzero"`
	// Echo includes the prefilled part of a continued assistant message.
	Echo *bool `json:"echo,omitzero"`

	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ChatContinuation is the shared [llama.Continuation] used by chat and template requests.
type ChatContinuation = llama.Continuation

const (
	// ChatContinuationUnset omits continue_final_message from a request.
	ChatContinuationUnset = llama.ContinuationUnset
	// ChatContinuationNull sends an explicit null.
	ChatContinuationNull = llama.ContinuationNull
	// ChatContinuationDisabled sends false.
	ChatContinuationDisabled = llama.ContinuationDisabled
	// ChatContinuationAuto sends true and lets the server select the field.
	ChatContinuationAuto = llama.ContinuationAuto
	// ChatContinuationContent continues the content field.
	ChatContinuationContent = llama.ContinuationContent
	// ChatContinuationReasoning continues the reasoning_content field.
	ChatContinuationReasoning = llama.ContinuationReasoning
)

// ChatCompletionResponse is the result of a non-streaming chat completion.
type ChatCompletionResponse struct {
	// ID identifies the completion.
	ID string `json:"id"`
	// Object is "chat.completion".
	Object string `json:"object"`
	// Created is a Unix timestamp in seconds.
	Created int64 `json:"created"`
	// Model identifies the responding model.
	Model string `json:"model"`
	// SystemFingerprint identifies the server build.
	SystemFingerprint string `json:"system_fingerprint"`
	// Choices contains one message per completion.
	Choices []ChatCompletionChoice `json:"choices"`
	// Usage counts prompt and generated tokens across the choices.
	Usage Usage `json:"usage"`
	// Timings reports llama.cpp evaluation statistics when available.
	Timings *llama.Timings `json:"timings,omitzero"`
	// Verbose is the native debug response, whose members depend on response_fields.
	Verbose jsontext.Value `json:"__verbose,omitzero"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ChatCompletionChoice is one complete assistant response.
type ChatCompletionChoice struct {
	// Index is the zero-based choice index.
	Index int `json:"index"`
	// Message is the assistant's reply, including reasoning or tool calls when produced.
	Message Message `json:"message"`
	// FinishReason is "length", "stop", or "tool_calls".
	FinishReason string `json:"finish_reason"`
	// Logprobs contains token probabilities if requested.
	Logprobs *ChatCompletionLogprobs `json:"logprobs,omitzero"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ChatCompletionLogprobs contains the server's per-token probabilities and alternatives.
type ChatCompletionLogprobs struct {
	// Content gives one distribution per generated token.
	Content []llama.TokenProb `json:"content"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ChatCompletionChunk is a streamed update. The final usage chunk has an empty Choices array.
type ChatCompletionChunk struct {
	// ID identifies the live completion for ControlChatCompletion.
	ID string `json:"id"`
	// Object is "chat.completion.chunk".
	Object string `json:"object"`
	// Created is a Unix timestamp in seconds.
	Created int64 `json:"created"`
	// Model identifies the responding model.
	Model string `json:"model"`
	// SystemFingerprint identifies the server build.
	SystemFingerprint string `json:"system_fingerprint"`
	// Choices identifies the choice receiving this update.
	Choices []ChatCompletionChunkChoice `json:"choices"`
	// Usage appears only in a final chunk requested with include_usage.
	Usage *Usage `json:"usage,omitzero"`
	// Timings reports final statistics or per-token timings if requested.
	Timings *llama.Timings `json:"timings,omitzero"`
	// PromptProgress reports prompt processing before token generation.
	PromptProgress *llama.PromptProgress `json:"prompt_progress,omitzero"`
	// Verbose is the native debug response, whose members depend on response_fields.
	Verbose jsontext.Value `json:"__verbose,omitzero"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ChatCompletionChunkChoice carries a partial assistant message or the finish reason with an empty delta.
type ChatCompletionChunkChoice struct {
	// Index identifies the choice to which Delta belongs.
	Index int `json:"index"`
	// Delta carries only the message fields changed by this chunk.
	Delta Message `json:"delta"`
	// FinishReason is null until the choice finishes.
	FinishReason *string `json:"finish_reason"`
	// Logprobs contains probabilities for the current token when available.
	Logprobs *ChatCompletionLogprobs `json:"logprobs,omitzero"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ControlChatCompletionRequest acts on a live completion by its streamed ID.
type ControlChatCompletionRequest struct {
	// ID is the identifier received on the open completion stream.
	ID string `json:"id"`
	// Action is "reasoning_end" to end the current reasoning block.
	Action string `json:"action"`
	// Model selects the instance holding the completion when using a router.
	Model string `json:"model,omitzero"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ControlChatCompletionResponse reports whether the action took effect; a finished ID is not an HTTP error.
type ControlChatCompletionResponse struct {
	// Success reports whether the server applied the action to a live completion.
	Success bool `json:"success"`
	// Message explains an unsuccessful action.
	Message *string `json:"message,omitzero"`
	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ChatCompletions generates complete assistant messages with POST /v1/chat/completions.
func (c *Client) ChatCompletions(ctx context.Context, req *ChatCompletionRequest, opts ...llama.RequestOption) (*ChatCompletionResponse, error) {
	if req != nil && req.Stream != nil && *req.Stream {
		r := *req
		r.Stream = new(false)
		req = &r
	}
	return llama.Call[*ChatCompletionResponse](ctx, c.c, http.MethodPost, chatCompletionsPath, req, opts...)
}

// ChatCompletionsStream streams assistant message deltas until the server's [DONE] marker without changing req.
// A stream that ends before the "[DONE]" marker yields an error matching llama.ErrIncompleteStream.
func (c *Client) ChatCompletionsStream(ctx context.Context, req *ChatCompletionRequest, opts ...llama.RequestOption) iter.Seq2[*ChatCompletionChunk, error] {
	var r ChatCompletionRequest
	if req != nil {
		r = *req
	}
	r.Stream = new(true)
	return llama.Stream[ChatCompletionChunk](ctx, c.c, http.MethodPost, chatCompletionsPath, &r, opts...)
}

// ControlChatCompletion sends a control action with POST /v1/chat/completions/control.
func (c *Client) ControlChatCompletion(ctx context.Context, req *ControlChatCompletionRequest, opts ...llama.RequestOption) (*ControlChatCompletionResponse, error) {
	return llama.Call[*ControlChatCompletionResponse](ctx, c.c, http.MethodPost, chatCompletionsPath+"/control", req, opts...)
}

// ChatInputTokens counts the tokens in a templated chat prompt without running generation.
func (c *Client) ChatInputTokens(ctx context.Context, req *ChatCompletionRequest, opts ...llama.RequestOption) (*InputTokensResponse, error) {
	return llama.Call[*InputTokensResponse](ctx, c.c, http.MethodPost, chatCompletionsPath+"/input_tokens", req, opts...)
}
