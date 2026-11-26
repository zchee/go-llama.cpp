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
	"maps"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
)

// ResponsesRequest is the input to POST /v1/responses and /v1/responses/input_tokens.
// llama.cpp converts it to a chat completion. Pass conversation history in Input;
// it does not store previous responses or resolve previous_response_id.
type ResponsesRequest struct {
	llama.SamplingParams
	Model             *string                   `json:"model,omitzero"`
	Input             ResponseInput             `json:"input"`
	Instructions      *string                   `json:"instructions,omitzero"`
	MaxOutputTokens   *int                      `json:"max_output_tokens,omitzero"`
	Stream            *bool                     `json:"stream,omitzero"`
	Tools             []ResponseTool            `json:"tools,omitzero"`
	ToolChoice        ToolChoice                `json:"tool_choice,omitzero"`
	ParallelToolCalls *bool                     `json:"parallel_tool_calls,omitzero"`
	Reasoning         *ResponseReasoningOptions `json:"reasoning,omitzero"`
	Text              *ResponseTextOptions      `json:"text,omitzero"`
	ExtraParams       *ResponseExtraParams      `json:"extra_params,omitzero"`
	Extra             map[string]jsontext.Value `json:",embed"`
}

// ResponseTool is a flat Responses function definition. Non-function tools are ignored by the server.
type ResponseTool struct {
	Type        string                    `json:"type"`
	Name        *string                   `json:"name,omitzero"`
	Description *string                   `json:"description,omitzero"`
	Parameters  jsontext.Value            `json:"parameters,omitzero"`
	Strict      *bool                     `json:"strict,omitzero"`
	Extra       map[string]jsontext.Value `json:",embed"`
}

// ResponseReasoningOptions configures the reasoning effort of the model.
type ResponseReasoningOptions struct {
	Effort *string                   `json:"effort,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// ResponseTextOptions configures the format of the response text.
type ResponseTextOptions struct {
	Format *ResponseTextFormat       `json:"format,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// ResponseTextFormat describes a text, json_object or json_schema format.
// Unlike the chat endpoint, the name and schema are members of this object.
type ResponseTextFormat struct {
	Type string `json:"type"`
	JSONSchemaFormat
}

// ResponseExtraParams overrides fields in the converted chat request, including sampling options.
type ResponseExtraParams struct {
	llama.SamplingParams
	Extra map[string]jsontext.Value `json:",embed"`
}

// Response is a completed response or a response snapshot in a lifecycle event.
// In-progress snapshots contain only ID, Object and Status.
type Response struct {
	ID          string                    `json:"id"`
	Object      string                    `json:"object"`
	Status      string                    `json:"status"`
	CreatedAt   *int64                    `json:"created_at,omitzero"`
	CompletedAt *int64                    `json:"completed_at,omitzero"`
	Model       *string                   `json:"model,omitzero"`
	Output      []ResponseItem            `json:"output,omitzero"`
	Usage       *ResponseUsage            `json:"usage,omitzero"`
	Extra       map[string]jsontext.Value `json:",embed"`
}

// ResponseUsage counts the tokens consumed and produced by the model.
type ResponseUsage struct {
	InputTokens        int64                     `json:"input_tokens"`
	OutputTokens       int64                     `json:"output_tokens"`
	TotalTokens        int64                     `json:"total_tokens"`
	InputTokensDetails llama.PromptTokensDetails `json:"input_tokens_details"`
	Extra              map[string]jsontext.Value `json:",embed"`
}

// Responses creates a non-streaming response. It never mutates req.
func (c *Client) Responses(ctx context.Context, req *ResponsesRequest, opts ...llama.RequestOption) (*Response, error) {
	return llama.Call[*Response](ctx, c.c, http.MethodPost, "/v1/responses", responsesStreamMode(req, false), opts...)
}

// ResponsesStream creates a streaming response. The sequence ends at EOF after response.completed;
// a stream that ends without a response.completed event yields an error matching llama.ErrIncompleteStream.
func (c *Client) ResponsesStream(ctx context.Context, req *ResponsesRequest, opts ...llama.RequestOption) iter.Seq2[*ResponseEvent, error] {
	return llama.Stream[ResponseEvent](ctx, c.c, http.MethodPost, "/v1/responses", responsesStreamMode(req, true), opts...)
}

// ResponsesInputTokens counts the tokens the converted request would use, without generating a response.
func (c *Client) ResponsesInputTokens(ctx context.Context, req *ResponsesRequest, opts ...llama.RequestOption) (*InputTokensResponse, error) {
	return llama.Call[*InputTokensResponse](ctx, c.c, http.MethodPost, "/v1/responses/input_tokens", req, opts...)
}

func responsesStreamMode(req *ResponsesRequest, stream bool) *ResponsesRequest {
	if req == nil {
		return nil
	}
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
