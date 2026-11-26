// Copyright 2026 The go-llama.cpp Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
)

const embeddingsPath = "/v1/embeddings"

// EmbeddingRequest requests embeddings from the OpenAI-compatible endpoint.
// Set Input. For legacy content-only requests, use [llama.Client.Embeddings] instead.
type EmbeddingRequest struct {
	Input llama.EmbeddingInput `json:"input,omitzero"`
	// Model selects a model in router mode and overrides the model name in the response.
	Model          *string                        `json:"model,omitzero"`
	EncodingFormat *llama.EmbeddingEncodingFormat `json:"encoding_format,omitzero"`
	// Normalize sets the llama.cpp normalization for the pooled vectors.
	Normalize *int `json:"embd_normalize,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingResponse is the list of embeddings and the total token usage.
type EmbeddingResponse struct {
	Object string            `json:"object"`
	Model  string            `json:"model"`
	Data   []EmbeddingResult `json:"data"`
	Usage  EmbeddingUsage    `json:"usage"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingResult is one pooled vector and its index in the response.
type EmbeddingResult struct {
	Object         string                         `json:"object"`
	Index          int                            `json:"index"`
	Embedding      EmbeddingVector                `json:"embedding"`
	EncodingFormat *llama.EmbeddingEncodingFormat `json:"encoding_format,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingUsage counts the input tokens for the whole request.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingVector is a numeric vector decoded from a float array or a base64 string.
type EmbeddingVector = llama.EmbeddingVector

// Embeddings sends a request to POST /v1/embeddings. The server must use pooled embeddings.
func (c *Client) Embeddings(ctx context.Context, req *EmbeddingRequest, opts ...llama.RequestOption) (*EmbeddingResponse, error) {
	if req != nil && req.Input.IsZero() {
		if _, legacy := req.Extra["content"]; legacy {
			return nil, errors.New("openai: use llama.Client.Embeddings for legacy content-only requests")
		}
	}
	return llama.Call[*EmbeddingResponse](ctx, c.c, http.MethodPost, embeddingsPath, req, opts...)
}
