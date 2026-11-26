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

package anthropic

import (
	"context"
	"encoding/json/jsontext"
	"iter"
	"maps"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
)

// MessagesRequest is the input to Messages, MessagesStream and CountTokens.
type MessagesRequest struct {
	// Model selects the model in router mode; a single-model server ignores it.
	Model *string `json:"model,omitzero"`
	// Messages is the conversation in chronological order.
	Messages []Message `json:"messages"`
	// System is a system instruction. For a block list the server concatenates text blocks.
	System Content `json:"system,omitzero"`
	// MaxTokens limits generated tokens and is not needed to count the prompt.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// Temperature controls randomness; zero selects greedy sampling.
	Temperature *float64 `json:"temperature,omitzero"`
	// TopP limits the cumulative probability of sampled tokens.
	TopP *float64 `json:"top_p,omitzero"`
	// TopK limits the number of candidate tokens.
	TopK *int `json:"top_k,omitzero"`
	// StopSequences are strings that stop generation; the server also accepts one string.
	StopSequences llama.StringOrSlice `json:"stop_sequences,omitzero"`
	// Stream selects the wire format. Messages and MessagesStream set it on a copy.
	Stream *bool `json:"stream,omitzero"`
	// Tools describes the callable functions and requires --jinja.
	Tools []Tool `json:"tools,omitzero"`
	// ToolChoice selects automatic or required tool use.
	ToolChoice *ToolChoice `json:"tool_choice,omitzero"`
	// IDSlot assigns the request to a server slot.
	IDSlot *int `json:"id_slot,omitzero"`
	// ChatTemplateKwargs supplies template-specific arguments.
	ChatTemplateKwargs map[string]jsontext.Value `json:"chat_template_kwargs,omitzero"`
	// Thinking sets the reasoning budget for a template that supports it.
	Thinking *Thinking `json:"thinking,omitzero"`
	// OutputConfig passes reasoning effort to the chat template.
	OutputConfig *OutputConfig `json:"output_config,omitzero"`
	// Metadata supplies user metadata during conversion.
	Metadata *Metadata `json:"metadata,omitzero"`
	// ExtraParams overrides converted chat parameters before inference.
	ExtraParams *ExtraParams `json:"extra_params,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ExtraParams holds llama.cpp-specific chat parameters inside extra_params, not at the request's top level.
type ExtraParams struct {
	llama.SamplingParams
	// Echo includes the assistant prefill in the parsed reply.
	Echo *bool `json:"echo,omitzero"`
	// ParallelToolCalls permits more than one tool call when the template supports it.
	ParallelToolCalls *bool                     `json:"parallel_tool_calls,omitzero"`
	Extra             map[string]jsontext.Value `json:",embed"`
}

// Tool defines a callable function.
type Tool struct {
	// Name identifies the function.
	Name string `json:"name"`
	// Description tells the model when to use the function.
	Description *string `json:"description,omitzero"`
	// InputSchema is the JSON schema of the function arguments.
	InputSchema jsontext.Value `json:"input_schema,omitzero"`
	// DeferLoading delays loading the tool until referenced by a tool result.
	DeferLoading *bool                     `json:"defer_loading,omitzero"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// ToolChoice selects how the model uses tools.
type ToolChoice struct {
	// Type is "auto", "any" or "tool". Both "any" and "tool" require a call; llama.cpp ignores a tool name here.
	Type  string                    `json:"type"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// Thinking supplies the Anthropic thinking mode and its optional token budget.
type Thinking struct {
	// Type must be "enabled" for the server to use BudgetTokens.
	Type string `json:"type"`
	// BudgetTokens limits the number of reasoning tokens.
	BudgetTokens *int                      `json:"budget_tokens,omitzero"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// OutputConfig controls reasoning effort for a template that supports it.
type OutputConfig struct {
	// Effort is the reasoning effort passed to the chat template.
	Effort *string                   `json:"effort,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// Metadata supplies user metadata to the request converter.
type Metadata struct {
	// UserID is the client-provided user identifier.
	UserID *string                   `json:"user_id,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// MessageResponse is the assistant's reply, also sent empty inside message_start.
type MessageResponse struct {
	// ID identifies the completion.
	ID string `json:"id"`
	// Type is "message".
	Type string `json:"type"`
	// Role is "assistant".
	Role string `json:"role"`
	// Content holds thinking, text and tool_use blocks in that order when present.
	Content []ContentBlock `json:"content"`
	// Model names the model that served the request.
	Model string `json:"model"`
	// StopReason is "end_turn", "max_tokens" or "tool_use", and null in message_start.
	StopReason *string `json:"stop_reason"`
	// StopSequence is the matched stop string, or null.
	StopSequence *string `json:"stop_sequence"`
	// Usage reports prompt and generation token counts.
	Usage Usage                     `json:"usage"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// Usage is the token accounting of an assistant message.
type Usage struct {
	// CacheReadInputTokens counts reused prompt tokens.
	CacheReadInputTokens int `json:"cache_read_input_tokens"`
	// InputTokens counts prompt tokens excluding cache hits.
	InputTokens int `json:"input_tokens"`
	// OutputTokens counts generated tokens.
	OutputTokens int                       `json:"output_tokens"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// CountTokensResponse reports the size of the prompt after applying the chat template.
type CountTokensResponse struct {
	// InputTokens includes all tokens of the rendered prompt.
	InputTokens int64                     `json:"input_tokens"`
	Extra       map[string]jsontext.Value `json:",embed"`
}

// withStream selects the format without changing the caller's request. The server merges extra_params last.
func (r *MessagesRequest) withStream(stream bool) *MessagesRequest {
	if r == nil {
		return nil
	}
	out := *r
	out.Stream = new(stream)
	if p := r.ExtraParams; p != nil {
		if _, ok := p.Extra["stream"]; ok {
			params := *p
			params.Extra = maps.Clone(p.Extra)
			delete(params.Extra, "stream")
			out.ExtraParams = &params
		}
	}
	return &out
}

// Messages generates one assistant reply with POST /v1/messages.
func (c *Client) Messages(ctx context.Context, req *MessagesRequest, opts ...llama.RequestOption) (*MessageResponse, error) {
	return llama.Call[*MessageResponse](ctx, c.c, http.MethodPost, "/v1/messages", req.withStream(false), opts...)
}

// MessagesStream generates a stream of named events with POST /v1/messages. The stream ends at EOF after
// message_stop, without a [DONE] frame. An error event yields an *llama.APIError and ends iteration.
func (c *Client) MessagesStream(ctx context.Context, req *MessagesRequest, opts ...llama.RequestOption) iter.Seq2[*MessageEvent, error] {
	return llama.Stream[MessageEvent](ctx, c.c, http.MethodPost, "/v1/messages", req.withStream(true), opts...)
}

// CountTokens counts the rendered prompt without generating a reply (POST /v1/messages/count_tokens).
func (c *Client) CountTokens(ctx context.Context, req *MessagesRequest, opts ...llama.RequestOption) (*CountTokensResponse, error) {
	return llama.Call[*CountTokensResponse](ctx, c.c, http.MethodPost, "/v1/messages/count_tokens", req, opts...)
}
