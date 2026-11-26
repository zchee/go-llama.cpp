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
	"fmt"
	"net/http"
)

// ApplyTemplateRequest is the body sent to POST /apply-template. The server parses the chat options but does not run inference.
type ApplyTemplateRequest struct {
	// Messages are the conversation turns, in order.
	Messages []Message `json:"messages,omitzero"`
	// AddGenerationPrompt appends the start of an assistant turn; the server defaults to true.
	AddGenerationPrompt *bool `json:"add_generation_prompt,omitzero"`
	// ContinueFinalMessage continues the final turn instead of starting a new one.
	ContinueFinalMessage ApplyTemplateContinuation `json:"continue_final_message,omitzero"`
	// ChatTemplateKwargs overrides the template variables configured on the server.
	ChatTemplateKwargs map[string]jsontext.Value `json:"chat_template_kwargs,omitzero"`
	// Tools contains OpenAI-format tool declarations; nonempty tools require Jinja.
	Tools []Tool `json:"tools,omitzero"`
	// ToolChoice is a mode, a named function, or a list of allowed tools.
	ToolChoice ToolChoice `json:"tool_choice,omitzero"`
	// ParallelToolCalls overrides the template's default for parallel tool calls.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitzero"`
	// ResponseFormat is the OpenAI-format object describing the requested output format.
	ResponseFormat *ResponseFormat `json:"response_format,omitzero"`
	// JSONSchema constrains the response format and cannot be combined with Grammar.
	JSONSchema jsontext.Value `json:"json_schema,omitzero"`
	// Grammar constrains the response via GBNF and cannot be combined with JSONSchema or active tools.
	Grammar *string `json:"grammar,omitzero"`
	// ReasoningFormat overrides the server's reasoning format.
	ReasoningFormat *string `json:"reasoning_format,omitzero"`
	// ReasoningEffort is a model-specific reasoning level, or "none" to disable reasoning.
	ReasoningEffort *string `json:"reasoning_effort,omitzero"`
	// Echo asks the chat parser to include the prompt when starting its session.
	Echo *bool `json:"echo,omitzero"`
	// Stop is one stop string or an array of stop strings. No generation runs here.
	Stop StringOrSlice `json:"stop,omitzero"`
	// Stream participates in the chat parser's validation; it does not stream the prompt response.
	Stream *bool `json:"stream,omitzero"`
	// Logprobs enables TopLogprobs in the chat parser; it produces no probabilities here.
	Logprobs *bool `json:"logprobs,omitzero"`
	// TopLogprobs requires Logprobs to be true.
	TopLogprobs *int `json:"top_logprobs,omitzero"`
	// ReasoningBudgetTokens limits reasoning tokens in the parsed chat options.
	ReasoningBudgetTokens *int `json:"reasoning_budget_tokens,omitzero"`
	// ThinkingBudgetTokens is an alias used when ReasoningBudgetTokens is omitted.
	ThinkingBudgetTokens *int `json:"thinking_budget_tokens,omitzero"`
	// ReasoningBudgetMessage is the message the chat parser assigns to the end of a reasoning budget.
	ReasoningBudgetMessage *string `json:"reasoning_budget_message,omitzero"`
	// ReasoningControl enables external control in the parsed chat options.
	ReasoningControl *bool `json:"reasoning_control,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ApplyTemplateContinuation represents a boolean, a named continuation mode, or explicit null.
// The zero value is omitted from ApplyTemplateRequest.
type ApplyTemplateContinuation string

const (
	// ApplyTemplateContinueAuto sends true and allows the template to select a continuation mode.
	ApplyTemplateContinueAuto ApplyTemplateContinuation = "true"
	// ApplyTemplateContinueDisabled sends false and disables explicit continuation.
	ApplyTemplateContinueDisabled ApplyTemplateContinuation = "false"
	// ApplyTemplateContinueContent continues the final message's content.
	ApplyTemplateContinueContent ApplyTemplateContinuation = "content"
	// ApplyTemplateContinueReasoning continues the final message's reasoning content.
	ApplyTemplateContinueReasoning ApplyTemplateContinuation = "reasoning_content"
	// ApplyTemplateContinueNull sends null and uses the server's default continuation behavior.
	ApplyTemplateContinueNull ApplyTemplateContinuation = "null"
)

// MarshalJSONTo implements json.MarshalerTo.
func (c ApplyTemplateContinuation) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch c {
	case "", ApplyTemplateContinueNull:
		return enc.WriteToken(jsontext.Null)
	case ApplyTemplateContinueAuto:
		return enc.WriteToken(jsontext.True)
	case ApplyTemplateContinueDisabled:
		return enc.WriteToken(jsontext.False)
	case ApplyTemplateContinueContent, ApplyTemplateContinueReasoning:
		return enc.WriteToken(jsontext.String(string(c)))
	default:
		return fmt.Errorf("llama: invalid template continuation mode %q", c)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (c *ApplyTemplateContinuation) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	switch kind := tok.Kind(); kind {
	case 'n':
		*c = ApplyTemplateContinueNull
	case 't':
		*c = ApplyTemplateContinueAuto
	case 'f':
		*c = ApplyTemplateContinueDisabled
	case '"':
		mode := ApplyTemplateContinuation(tok.String())
		if mode != ApplyTemplateContinueContent && mode != ApplyTemplateContinueReasoning {
			return fmt.Errorf("llama: invalid template continuation mode %q", mode)
		}
		*c = mode
	default:
		return fmt.Errorf("llama: template continuation must be a boolean, string or null, got %s", kind)
	}
	return nil
}

// ApplyTemplateResponse is the JSON returned by POST /apply-template.
type ApplyTemplateResponse struct {
	// Prompt is the rendered conversation, without running inference.
	Prompt string `json:"prompt"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ApplyTemplate renders a conversation with the server's configured chat template.
func (c *Client) ApplyTemplate(ctx context.Context, req *ApplyTemplateRequest, opts ...RequestOption) (*ApplyTemplateResponse, error) {
	return Call[*ApplyTemplateResponse](ctx, c, http.MethodPost, "/apply-template", req, opts...)
}
