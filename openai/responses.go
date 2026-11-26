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
	"maps"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/streamerr"
)

// ResponsesRequest is the input to POST /v1/responses and /v1/responses/input_tokens.
// llama.cpp converts it to a chat completion. Pass conversation history in Input;
// it does not store previous responses or resolve previous_response_id.
type ResponsesRequest struct {
	llama.SamplingParams
	// Model selects the target model in router mode; single-model servers ignore it.
	Model string `json:"model,omitzero"`
	// Input supplies the required user prompt or ordered conversation history.
	Input ResponseInput `json:"input"`
	// Instructions prepends a system message to Input; omission adds no system message.
	Instructions *string `json:"instructions,omitzero"`
	// MaxOutputTokens caps generated tokens as max_tokens in the converted chat request, which uses the server limit
	// when no token limit is supplied.
	MaxOutputTokens *int `json:"max_output_tokens,omitzero"`
	// Stream selects SSE instead of a single JSON response; Responses and ResponsesStream set it automatically.
	Stream *bool `json:"stream,omitzero"`
	// Tools defines the functions available to the model; omission advertises no functions.
	Tools []ResponseTool `json:"tools,omitzero"`
	// ToolChoice controls when and which functions may be called, defaulting to "auto".
	ToolChoice ToolChoice `json:"tool_choice,omitzero"`
	// ParallelToolCalls allows more than one tool call in a reply, defaulting to the chat template's capability.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitzero"`
	// Reasoning overrides the chat template's reasoning effort when provided.
	Reasoning *ResponseReasoningOptions `json:"reasoning,omitzero"`
	// Text sets an output text constraint; omission adds no constraint.
	Text *ResponseTextOptions `json:"text,omitzero"`
	// ExtraParams applies last to the converted chat request and leaves it unchanged when omitted.
	ExtraParams *ResponseExtraParams `json:"extra_params,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseTool is a flat Responses function definition. Non-function tools are ignored by the server.
type ResponseTool struct {
	// Type must be "function" for the server to advertise this tool to the model.
	Type string `json:"type"`
	// Name is the required function name the model uses in a call.
	Name *string `json:"name,omitzero"`
	// Description tells the model when to call the function; omission supplies no description.
	Description *string `json:"description,omitzero"`
	// Parameters is the JSON Schema for the function's arguments, defaulting to an empty object.
	Parameters jsontext.Value `json:"parameters,omitzero"`
	// Strict defaults to true during conversion to chat, but the chat tool parser ignores it.
	Strict *bool `json:"strict,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseReasoningOptions configures the reasoning effort of the model.
type ResponseReasoningOptions struct {
	// Effort sets the chat template's reasoning level, with "none" disabling reasoning and omission using the default.
	Effort *string `json:"effort,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseTextOptions configures the format of the response text.
type ResponseTextOptions struct {
	// Format selects text or a JSON constraint; omission leaves the chat response format unchanged.
	Format *ResponseTextFormat `json:"format,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseTextFormat describes a text, json_object or json_schema format.
// Unlike the chat endpoint, the name and schema are members of this object.
type ResponseTextFormat struct {
	// Type selects unconstrained "text", "json_object", or "json_schema"; an empty value leaves the format unchanged.
	Type string `json:"type"`
	JSONSchemaFormat
}

// ResponseExtraParams overrides fields in the converted chat request, including sampling options.
type ResponseExtraParams struct {
	llama.SamplingParams
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Response is a completed response or a response snapshot in a lifecycle event.
// In-progress snapshots contain only ID, Object and Status.
type Response struct {
	// ID is the server-assigned response identifier shared across lifecycle events.
	ID string `json:"id"`
	// Object is "response" in every snapshot and in the non-streaming response.
	Object string `json:"object"`
	// Status is "in_progress" in the initial stream snapshots and "completed" in the final response.
	Status string `json:"status"`
	// CreatedAt is the Unix timestamp in seconds, written when the server builds the completed response.
	CreatedAt *int64 `json:"created_at,omitzero"`
	// CompletedAt is the Unix timestamp in seconds for completion and appears only on non-streaming responses.
	CompletedAt *int64 `json:"completed_at,omitzero"`
	// Model identifies the model that generated the completed response and is absent from in-progress snapshots.
	Model *string `json:"model,omitzero"`
	// Output contains the generated reasoning, message, and tool calls in order and is absent while in progress.
	Output []ResponseItem `json:"output,omitzero"`
	// Usage reports the token totals for a completed response and is absent while in progress.
	Usage *ResponseUsage `json:"usage,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseUsage counts the tokens consumed and produced by the model.
type ResponseUsage struct {
	// InputTokens counts the tokens in the converted, templated chat prompt.
	InputTokens int64 `json:"input_tokens"`
	// OutputTokens counts tokens decoded by the model, including any reasoning and function calls.
	OutputTokens int64 `json:"output_tokens"`
	// TotalTokens is the sum of InputTokens and OutputTokens.
	TotalTokens int64 `json:"total_tokens"`
	// InputTokensDetails counts how many prompt tokens the server reused from its cache.
	InputTokensDetails llama.PromptTokensDetails `json:"input_tokens_details"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Responses creates a non-streaming response. It never mutates req.
func (c *Client) Responses(ctx context.Context, req *ResponsesRequest, opts ...llama.RequestOption) (*Response, error) {
	if req == nil {
		return nil, errors.New("openai: nil responses request")
	}
	return llama.Call[*Response](ctx, c.c, http.MethodPost, "/v1/responses", responsesStreamMode(req, false), opts...)
}

// ResponsesStream creates a streaming response. The sequence ends at EOF after response.completed;
// a stream that ends without a response.completed event yields an error matching llama.ErrIncompleteStream.
func (c *Client) ResponsesStream(ctx context.Context, req *ResponsesRequest, opts ...llama.RequestOption) iter.Seq2[*ResponseEvent, error] {
	if req == nil {
		return streamerr.Once[ResponseEvent](errors.New("openai: nil responses stream request"), llama.ErrStreamConsumed)
	}
	return llama.Stream[ResponseEvent](ctx, c.c, http.MethodPost, "/v1/responses", responsesStreamMode(req, true), opts...)
}

// ResponsesInputTokens counts the tokens the converted request would use, without generating a response.
func (c *Client) ResponsesInputTokens(ctx context.Context, req *ResponsesRequest, opts ...llama.RequestOption) (*InputTokensResponse, error) {
	if req == nil {
		return nil, errors.New("openai: nil responses input tokens request")
	}
	return llama.Call[*InputTokensResponse](ctx, c.c, http.MethodPost, "/v1/responses/input_tokens", req, opts...)
}

func responsesStreamMode(req *ResponsesRequest, stream bool) *ResponsesRequest {
	r := *req
	r.Stream = new(stream)
	// The server merges extra_params last, even when it replaces stream.
	if r.ExtraParams != nil {
		if _, ok := r.ExtraParams.Extra["stream"]; ok {
			p := *r.ExtraParams
			p.Extra = maps.Clone(p.Extra)
			delete(p.Extra, "stream")
			r.ExtraParams = &p
		}
	}
	return &r
}
