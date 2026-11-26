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

package llama

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"net/http"
)

// ApplyTemplateRequest is the body sent to POST /apply-template. The server renders a prompt without running inference.
// It parses ReasoningBudgetTokens, ThinkingBudgetTokens, ReasoningBudgetMessage and ReasoningControl
// for generation, but none of them affects the rendered prompt.
type ApplyTemplateRequest struct {
	// Messages supplies the required conversation array in turn order.
	Messages []Message `json:"messages,omitzero"`
	// AddGenerationPrompt appends the start of an assistant turn; defaults to true unless the server prefills it.
	AddGenerationPrompt *bool `json:"add_generation_prompt,omitzero"`
	// ContinueFinalMessage selects which final assistant field to continue; an explicit selection requires
	// AddGenerationPrompt to be false.
	ContinueFinalMessage Continuation `json:"continue_final_message,omitzero"`
	// ChatTemplateKwargs overrides individual server-configured template variables; omission keeps those defaults.
	ChatTemplateKwargs map[string]jsontext.Value `json:"chat_template_kwargs,omitzero"`
	// Tools supplies optional OpenAI-format tool declarations; a nonempty list requires Jinja.
	Tools []Tool `json:"tools,omitzero"`
	// ToolChoice is a mode, a named function, or a list of allowed tools; omission defaults to "auto".
	ToolChoice ToolChoice `json:"tool_choice,omitzero"`
	// ParallelToolCalls overrides the active template's capability-based default for parallel calls.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitzero"`
	// ResponseFormat supplies OpenAI-compatible format constraints to the chat parser; optional.
	ResponseFormat *ResponseFormat `json:"response_format,omitzero"`
	// JSONSchema constrains the response format and cannot be combined with Grammar.
	JSONSchema jsontext.Value `json:"json_schema,omitzero"`
	// Grammar constrains the response via GBNF and cannot be combined with JSONSchema or active tools.
	Grammar *string `json:"grammar,omitzero"`
	// ReasoningFormat overrides the server's reasoning mode for rendering the conversation.
	ReasoningFormat *string `json:"reasoning_format,omitzero"`
	// ReasoningEffort selects a template-specific level, or "none" to disable reasoning; defaults to no override.
	ReasoningEffort *string `json:"reasoning_effort,omitzero"`
	// Echo changes how the parser handles continued assistant content; defaults to false.
	Echo *bool `json:"echo,omitzero"`
	// Stop supplies one or more stop strings to the parser without truncating the rendered prompt.
	Stop StringOrSlice `json:"stop,omitzero"`
	// Stream participates in chat parser validation (default false); it does not stream the prompt response.
	Stream *bool `json:"stream,omitzero"`
	// Logprobs enables parsing of TopLogprobs (default false); it produces no probabilities here.
	Logprobs *bool `json:"logprobs,omitzero"`
	// TopLogprobs is the candidate count for the parsed request; requires Logprobs and defaults to 20 when enabled.
	TopLogprobs *int `json:"top_logprobs,omitzero"`
	// ReasoningBudgetTokens limits reasoning tokens in the parsed chat options; -1 uses the server's budget.
	ReasoningBudgetTokens *int `json:"reasoning_budget_tokens,omitzero"`
	// ThinkingBudgetTokens is an alias used when ReasoningBudgetTokens is omitted.
	ThinkingBudgetTokens *int `json:"thinking_budget_tokens,omitzero"`
	// ReasoningBudgetMessage sets the parsed text for the end of a reasoning budget; nil uses the server's text.
	ReasoningBudgetMessage *string `json:"reasoning_budget_message,omitzero"`
	// ReasoningControl enables external control in the parsed chat options; defaults to false.
	ReasoningControl *bool `json:"reasoning_control,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ApplyTemplateResponse is the JSON returned by POST /apply-template.
type ApplyTemplateResponse struct {
	// Prompt is the rendered conversation, without running inference.
	Prompt string `json:"prompt"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ApplyTemplate renders a conversation with the server's configured chat template.
func (c *Client) ApplyTemplate(ctx context.Context, req *ApplyTemplateRequest, opts ...RequestOption) (*ApplyTemplateResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil apply template request")
	}
	return Call[*ApplyTemplateResponse](ctx, c, http.MethodPost, "/apply-template", req, opts...)
}
