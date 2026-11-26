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
	"net/http"
)

// PropsResponse describes the active model or, without a model query in router mode, the router itself.
type PropsResponse struct {
	DefaultGenerationSettings PropsGenerationSettings   `json:"default_generation_settings"`
	ModelAlias                string                    `json:"model_alias"`
	ModelPath                 string                    `json:"model_path"`
	UISettings                map[string]jsontext.Value `json:"ui_settings"`
	BuildInfo                 string                    `json:"build_info"`
	CORSProxyEnabled          bool                      `json:"cors_proxy_enabled"`
	UIAgent                   bool                      `json:"ui_agent"`
	ConnectCode               *string                   `json:"connect_code,omitzero"`
	Role                      *string                   `json:"role,omitzero"`
	MaxInstances              *int                      `json:"max_instances,omitzero"`
	ModelsAutoload            *bool                     `json:"models_autoload,omitzero"`
	TotalSlots                *int                      `json:"total_slots,omitzero"`
	ModelFType                *string                   `json:"model_ftype,omitzero"`
	Modalities                *PropsModalities          `json:"modalities,omitzero"`
	MediaMarker               *string                   `json:"media_marker,omitzero"`
	EndpointSlots             *bool                     `json:"endpoint_slots,omitzero"`
	EndpointProps             *bool                     `json:"endpoint_props,omitzero"`
	EndpointMetrics           *bool                     `json:"endpoint_metrics,omitzero"`
	UI                        *bool                     `json:"ui,omitzero"`
	ChatTemplate              *string                   `json:"chat_template,omitzero"`
	ChatTemplateToolUse       *string                   `json:"chat_template_tool_use,omitzero"`
	ChatTemplateCaps          *PropsChatTemplateCaps    `json:"chat_template_caps,omitzero"`
	BOSToken                  *string                   `json:"bos_token,omitzero"`
	EOSToken                  *string                   `json:"eos_token,omitzero"`
	IsSleeping                *bool                     `json:"is_sleeping,omitzero"`
	Voices                    []string                  `json:"voices,omitzero"`
	Extra                     map[string]jsontext.Value `json:",embed"`
}

// PropsGenerationSettings gives the context size per slot and model defaults. Params is null for the router itself.
type PropsGenerationSettings struct {
	NCtx   int                       `json:"n_ctx"`
	Params *GenerationSettings       `json:"params"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// PropsModalities reports which kinds of media the model accepts.
type PropsModalities struct {
	Vision bool                      `json:"vision"`
	Audio  bool                      `json:"audio"`
	Video  bool                      `json:"video"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// PropsChatTemplateCaps reports input forms and reasoning effort levels supported by the model's chat template.
type PropsChatTemplateCaps struct {
	SupportsObjectArguments   bool                      `json:"supports_object_arguments"`
	SupportsParallelToolCalls bool                      `json:"supports_parallel_tool_calls"`
	SupportsPreserveReasoning bool                      `json:"supports_preserve_reasoning"`
	SupportsReasoningEffort   bool                      `json:"supports_reasoning_effort"`
	SupportsStringContent     bool                      `json:"supports_string_content"`
	SupportsSystemRole        bool                      `json:"supports_system_role"`
	SupportsToolCalls         bool                      `json:"supports_tool_calls"`
	SupportsTools             bool                      `json:"supports_tools"`
	SupportsTypedContent      bool                      `json:"supports_typed_content"`
	ReasoningEffortLevels     []string                  `json:"reasoning_effort_levels"`
	ReasoningEffortRejected   []string                  `json:"reasoning_effort_rejected"`
	Extra                     map[string]jsontext.Value `json:",embed"`
}

// PropsSetRequest addresses a model in router mode. The model server currently ignores the request body.
type PropsSetRequest struct {
	Model *string                   `json:"model,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// PropsSetResponse acknowledges a POST /props on a server started with --props.
type PropsSetResponse struct {
	Success bool                      `json:"success"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// Props fetches the properties of the server. WithModelQuery selects a model on a router.
func (c *Client) Props(ctx context.Context, opts ...RequestOption) (*PropsResponse, error) {
	return Call[*PropsResponse](ctx, c, http.MethodGet, "/props", nil, opts...)
}

// SetProps posts global properties. The server acknowledges the request but does not change any settings at this tag.
func (c *Client) SetProps(ctx context.Context, req *PropsSetRequest, opts ...RequestOption) (*PropsSetResponse, error) {
	return Call[*PropsSetResponse](ctx, c, http.MethodPost, "/props", req, opts...)
}
