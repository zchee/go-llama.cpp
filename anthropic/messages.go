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
	"errors"
	"iter"
	"maps"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/streamerr"
)

// MessagesRequest is the input to Messages, MessagesStream and CountTokens.
type MessagesRequest struct {
	// Model selects the model in router mode; a single-model server ignores it.
	Model string `json:"model,omitzero"`
	// Messages is the required conversation history in chronological order.
	Messages []Message `json:"messages"`
	// System optionally prepends an instruction; the server concatenates only the text blocks of a list.
	System Content `json:"system,omitzero"`
	// MaxTokens caps generation at this many tokens (4096 if omitted) and has no effect on prompt counting.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// Temperature controls sampling randomness; zero selects greedy sampling and omission uses the server setting.
	Temperature *float64 `json:"temperature,omitzero"`
	// TopP sets the cumulative probability cutoff from 0 to 1; omission uses the server setting.
	TopP *float64 `json:"top_p,omitzero"`
	// TopK caps the number of candidate tokens; zero disables the cap and omission uses the server setting.
	TopK *int `json:"top_k,omitzero"`
	// StopSequences optionally stops generation before emitting the first matched stop string.
	StopSequences llama.StringOrSlice `json:"stop_sequences,omitzero"`
	// Stream selects SSE instead of a single JSON reply; Messages and MessagesStream set it on a copy.
	Stream *bool `json:"stream,omitzero"`
	// Tools describes the callable functions and requires --jinja when nonempty.
	Tools []Tool `json:"tools,omitzero"`
	// ToolChoice selects automatic or required tool use, defaulting to automatic.
	ToolChoice *ToolChoice `json:"tool_choice,omitzero"`
	// IDSlot selects a server slot; the server chooses one when omitted or set to -1.
	IDSlot *int `json:"id_slot,omitzero"`
	// ChatTemplateKwargs supplies optional template-specific arguments.
	ChatTemplateKwargs map[string]jsontext.Value `json:"chat_template_kwargs,omitzero"`
	// Thinking optionally overrides the server's reasoning token budget when its type is "enabled".
	Thinking *Thinking `json:"thinking,omitzero"`
	// OutputConfig optionally passes reasoning effort to the chat template.
	OutputConfig *OutputConfig `json:"output_config,omitzero"`
	// Metadata optionally supplies a user identifier to the request converter.
	Metadata *Metadata `json:"metadata,omitzero"`
	// ExtraParams overrides converted chat parameters before inference; omission leaves them unchanged.
	ExtraParams *ExtraParams `json:"extra_params,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ExtraParams holds llama.cpp-specific chat parameters inside extra_params, not at the request's top level.
type ExtraParams struct {
	llama.SamplingParams
	// Echo includes the assistant prefill in the parsed reply; the default is false.
	Echo *bool `json:"echo,omitzero"`
	// ParallelToolCalls permits multiple tool calls, defaulting to whether the chat template supports them.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Tool defines a callable function.
type Tool struct {
	// Name identifies the function in calls and tool references.
	Name string `json:"name"`
	// Description optionally tells the model when to use the function.
	Description *string `json:"description,omitzero"`
	// InputSchema describes the function arguments as JSON Schema; the converter uses an empty object when omitted.
	InputSchema jsontext.Value `json:"input_schema,omitzero"`
	// DeferLoading hides the tool until a tool result references it; the default loads it immediately.
	DeferLoading *bool `json:"defer_loading,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolChoice selects how the model uses tools.
type ToolChoice struct {
	// Type defaults to "auto"; "any" and "tool" both require a call without selecting a named tool.
	Type string `json:"type"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Thinking supplies the Anthropic thinking mode and its optional token budget.
type Thinking struct {
	// Type must be "enabled" for the server to use BudgetTokens.
	Type string `json:"type"`
	// BudgetTokens limits reasoning tokens and defaults to 10000 when Type is "enabled".
	BudgetTokens *int `json:"budget_tokens,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// OutputConfig controls reasoning effort for a template that supports it.
type OutputConfig struct {
	// Effort overrides the template's reasoning effort when nonempty; "none" disables reasoning.
	Effort *string `json:"effort,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Metadata supplies user metadata to the request converter.
type Metadata struct {
	// UserID is forwarded as internal request metadata only when nonempty.
	UserID *string `json:"user_id,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
	// Usage reports prompt and generation token counts; generation is zero in message_start.
	Usage Usage `json:"usage"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Usage is the token accounting of an assistant message.
type Usage struct {
	// CacheReadInputTokens counts reused prompt tokens.
	CacheReadInputTokens int `json:"cache_read_input_tokens"`
	// InputTokens counts prompt tokens excluding cache hits.
	InputTokens int `json:"input_tokens"`
	// OutputTokens counts generated tokens.
	OutputTokens int `json:"output_tokens"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// CountTokensResponse reports the size of the prompt after applying the chat template.
type CountTokensResponse struct {
	// InputTokens includes all tokens of the rendered prompt.
	InputTokens int64 `json:"input_tokens"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// withStream selects the format without changing the caller's request. The server merges extra_params last.
func (r *MessagesRequest) withStream(stream bool) *MessagesRequest {
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
	if req == nil {
		return nil, errors.New("anthropic: nil messages request")
	}
	return llama.Call[*MessageResponse](ctx, c.c, http.MethodPost, "/v1/messages", req.withStream(false), opts...)
}

// MessagesStream generates a stream of named events with POST /v1/messages. The stream ends at EOF after
// message_stop, without a [DONE] frame. An error event yields an *llama.APIError and ends iteration.
// A stream that ends without a message_stop event yields an error matching llama.ErrIncompleteStream.
func (c *Client) MessagesStream(ctx context.Context, req *MessagesRequest, opts ...llama.RequestOption) iter.Seq2[*MessageEvent, error] {
	if req == nil {
		return streamerr.Once[MessageEvent](errors.New("anthropic: nil messages stream request"), llama.ErrStreamConsumed)
	}
	return llama.Stream[MessageEvent](ctx, c.c, http.MethodPost, "/v1/messages", req.withStream(true), opts...)
}

// CountTokens counts the rendered prompt without generating a reply (POST /v1/messages/count_tokens).
func (c *Client) CountTokens(ctx context.Context, req *MessagesRequest, opts ...llama.RequestOption) (*CountTokensResponse, error) {
	if req == nil {
		return nil, errors.New("anthropic: nil count tokens request")
	}
	return llama.Call[*CountTokensResponse](ctx, c.c, http.MethodPost, "/v1/messages/count_tokens", req, opts...)
}
