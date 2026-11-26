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

const loraAdaptersPath = "/lora-adapters"

// LoRAAdapter describes a loaded LoRA adapter and its global scale.
type LoRAAdapter struct {
	// ID identifies the adapter in subsequent requests.
	ID int `json:"id"`
	// Path is the adapter path on the server, not on the client.
	Path string `json:"path"`
	// Scale is the global scale; zero disables the adapter.
	Scale float64 `json:"scale"`
	// TaskName and PromptPrefix come from the adapter's metadata.
	TaskName     string `json:"task_name"`
	PromptPrefix string `json:"prompt_prefix"`
	// ALoRAInvocationString is present only for adapters with invocation tokens.
	ALoRAInvocationString *string `json:"alora_invocation_string,omitzero"`
	// ALoRAInvocationTokens are the token IDs that activate an aLoRA adapter.
	ALoRAInvocationTokens []Token `json:"alora_invocation_tokens,omitzero"`
	// Extra preserves unrecognized response members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// LoRASetAdaptersResponse reports whether the server updated its global scales.
type LoRASetAdaptersResponse struct {
	// Success reports whether the update was applied.
	Success bool `json:"success"`
	// Extra preserves unrecognized response members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// LoRAAdapters lists all loaded adapters. In router mode, select a model with WithModelQuery.
func (c *Client) LoRAAdapters(ctx context.Context, opts ...RequestOption) ([]LoRAAdapter, error) {
	adapters, err := Call[[]LoRAAdapter](ctx, c, http.MethodGet, loraAdaptersPath, nil, opts...)
	return adapters, c.RedactError(err)
}

// SetLoRAAdapters replaces the global adapter scales. Omitted adapters are disabled, so an empty
// or nil slice disables them all. A generation request can override the global scales for that request.
// In router mode, select a model with WithModelQuery.
func (c *Client) SetLoRAAdapters(ctx context.Context, weights []LoRAWeight, opts ...RequestOption) (*LoRASetAdaptersResponse, error) {
	res, err := Call[*LoRASetAdaptersResponse](ctx, c, http.MethodPost, loraAdaptersPath, weights, opts...)
	return res, c.RedactError(err)
}
