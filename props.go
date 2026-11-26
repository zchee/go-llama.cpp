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

// PropsResponse describes the active model or, without a model query in router mode, the router itself.
type PropsResponse struct {
	// DefaultGenerationSettings reports the model's sampling defaults and per-slot context size.
	DefaultGenerationSettings PropsGenerationSettings `json:"default_generation_settings"`
	// ModelAlias is the name advertised for the model, or "llama-server" for the router itself.
	ModelAlias string `json:"model_alias"`
	// ModelPath is the server-side model file path, or "none" for the router itself.
	ModelPath string `json:"model_path"`
	// UISettings carries server-configured defaults for the web UI as raw JSON.
	UISettings jsontext.Value `json:"ui_settings,omitzero"`
	// BuildInfo identifies the running server build by build number and commit.
	BuildInfo string `json:"build_info"`
	// CORSProxyEnabled reports whether the web UI's MCP CORS proxy is enabled.
	CORSProxyEnabled bool `json:"cors_proxy_enabled"`
	// UIAgent reports whether browser-only agent mode was enabled with --ui-agent.
	UIAgent bool `json:"ui_agent"`
	// ConnectCode is the pairing code used by --connect; omitted when none is configured.
	ConnectCode *string `json:"connect_code,omitzero"`
	// Role is "router" for the router itself and is omitted in model responses.
	Role *string `json:"role,omitzero"`
	// MaxInstances is the router's simultaneous model limit; 0 is unlimited and model responses omit it.
	MaxInstances *int `json:"max_instances,omitzero"`
	// ModelsAutoload reports whether the router loads models on demand; omitted in model responses.
	ModelsAutoload *bool `json:"models_autoload,omitzero"`
	// TotalSlots is the number of parallel processing slots; omitted for the router itself.
	TotalSlots *int `json:"total_slots,omitzero"`
	// ModelFType names the model's file precision or quantization type; omitted for the router itself.
	ModelFType *string `json:"model_ftype,omitzero"`
	// Modalities reports which kinds of media the loaded model accepts; omitted for the router itself.
	Modalities *PropsModalities `json:"modalities,omitzero"`
	// MediaMarker is the placeholder for each item in native multimodal prompts; omitted for the router itself.
	MediaMarker *string `json:"media_marker,omitzero"`
	// EndpointSlots reports whether GET /slots is enabled; omitted for the router itself.
	EndpointSlots *bool `json:"endpoint_slots,omitzero"`
	// EndpointProps reports whether POST /props is enabled; omitted for the router itself.
	EndpointProps *bool `json:"endpoint_props,omitzero"`
	// EndpointMetrics reports whether GET /metrics is enabled; omitted for the router itself.
	EndpointMetrics *bool `json:"endpoint_metrics,omitzero"`
	// UI reports whether the server serves the web interface; omitted for the router itself.
	UI *bool `json:"ui,omitzero"`
	// ChatTemplate is the source of the active default chat template; omitted for the router itself.
	ChatTemplate *string `json:"chat_template,omitzero"`
	// ChatTemplateToolUse gives the alternate tool-use template, when available and Jinja is enabled.
	ChatTemplateToolUse *string `json:"chat_template_tool_use,omitzero"`
	// ChatTemplateCaps reports the active template's supported input forms; omitted for the router itself.
	ChatTemplateCaps *PropsChatTemplateCaps `json:"chat_template_caps,omitzero"`
	// BOSToken is the spelling of the model's beginning-of-sequence token; omitted for the router itself.
	BOSToken *string `json:"bos_token,omitzero"`
	// EOSToken is the spelling of the model's end-of-sequence token; omitted for the router itself.
	EOSToken *string `json:"eos_token,omitzero"`
	// IsSleeping reports whether the model entered idle sleep; omitted for the router itself.
	IsSleeping *bool `json:"is_sleeping,omitzero"`
	// Voices lists predefined voice names accepted by /tts; omitted when the model offers none.
	Voices []string `json:"voices,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PropsGenerationSettings gives the context size per slot and model defaults. Params is null for the router itself.
type PropsGenerationSettings struct {
	// NCtx is the per-slot context capacity in tokens, or 0 for the router itself.
	NCtx int `json:"n_ctx"`
	// Params holds the reduced set of default generation settings; nil for the router itself.
	Params *GenerationSettings `json:"params"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PropsModalities reports which kinds of media the model accepts.
type PropsModalities struct {
	// Vision reports whether the model accepts image input.
	Vision bool `json:"vision"`
	// Audio reports whether the model accepts audio input, not whether it can synthesize speech.
	Audio bool `json:"audio"`
	// Video reports whether the model accepts video input.
	Video bool `json:"video"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PropsChatTemplateCaps reports input forms and reasoning effort levels supported by the model's chat template.
type PropsChatTemplateCaps struct {
	// SupportsObjectArguments reports whether the template takes tool call arguments as objects, not JSON text.
	SupportsObjectArguments bool `json:"supports_object_arguments"`
	// SupportsParallelToolCalls reports whether the template can render multiple tool calls in one assistant turn.
	SupportsParallelToolCalls bool `json:"supports_parallel_tool_calls"`
	// SupportsPreserveReasoning reports whether the template can retain reasoning from earlier assistant turns.
	SupportsPreserveReasoning bool `json:"supports_preserve_reasoning"`
	// SupportsReasoningEffort reports whether the template handles a reasoning effort setting.
	SupportsReasoningEffort bool `json:"supports_reasoning_effort"`
	// SupportsStringContent reports whether the template accepts plain strings as message content.
	SupportsStringContent bool `json:"supports_string_content"`
	// SupportsSystemRole reports whether the template handles system messages directly.
	SupportsSystemRole bool `json:"supports_system_role"`
	// SupportsToolCalls reports whether the template handles assistant tool calls in the conversation history.
	SupportsToolCalls bool `json:"supports_tool_calls"`
	// SupportsTools reports whether the template can render tool definitions in the prompt.
	SupportsTools bool `json:"supports_tools"`
	// SupportsTypedContent reports whether the template accepts message content as a list of typed parts.
	SupportsTypedContent bool `json:"supports_typed_content"`
	// ReasoningEffortLevels lists levels the template renders distinctly, from lowest to highest.
	ReasoningEffortLevels []string `json:"reasoning_effort_levels"`
	// ReasoningEffortRejected lists levels the template refuses with HTTP 400.
	ReasoningEffortRejected []string `json:"reasoning_effort_rejected"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PropsSetRequest addresses a model in router mode. The model server currently ignores the request body.
type PropsSetRequest struct {
	// Model identifies the target when sending through a router; ignored by the model server.
	Model string `json:"model,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PropsSetResponse acknowledges a POST /props on a server started with --props.
type PropsSetResponse struct {
	// Success is true when the server acknowledges the request, even though no setting is changed.
	Success bool `json:"success"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Props fetches the properties of the server. WithModelQuery selects a model on a router.
func (c *Client) Props(ctx context.Context, opts ...RequestOption) (*PropsResponse, error) {
	return Call[*PropsResponse](ctx, c, http.MethodGet, "/props", nil, opts...)
}

// SetProps posts global properties. The server acknowledges the request but does not change any settings at this tag.
func (c *Client) SetProps(ctx context.Context, req *PropsSetRequest, opts ...RequestOption) (*PropsSetResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil set props request")
	}
	return Call[*PropsSetResponse](ctx, c, http.MethodPost, "/props", req, opts...)
}
