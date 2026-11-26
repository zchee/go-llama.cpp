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
	// Input supplies a nonempty text, token, or multimodal prompt or batch of prompts to embed.
	Input llama.EmbeddingInput `json:"input,omitzero"`
	// Model selects a model in router mode and overrides the model name in the response.
	Model string `json:"model,omitzero"`
	// EncodingFormat selects "float" (the default) or "base64" for the returned vectors.
	EncodingFormat *llama.EmbeddingEncodingFormat `json:"encoding_format,omitzero"`
	// Normalize uses the server default when omitted; -1 disables normalization, 0 uses the maximum absolute value,
	// and positive p uses the Lp norm.
	Normalize *int `json:"embd_normalize,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingResponse is the list of embeddings and the total token usage.
type EmbeddingResponse struct {
	// Object is "list" for the response envelope.
	Object string `json:"object"`
	// Model echoes the requested model name or reports the server's model name if none was requested.
	Model string `json:"model"`
	// Data contains one pooled embedding per input in request order.
	Data []EmbeddingResult `json:"data"`
	// Usage counts the tokens evaluated across all inputs in the request.
	Usage EmbeddingUsage `json:"usage"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingResult is one pooled vector and its index in the response.
type EmbeddingResult struct {
	// Object is "embedding" for each entry in the list.
	Object string `json:"object"`
	// Index is the zero-based position of the input that produced this vector.
	Index int `json:"index"`
	// Embedding is a float array or a base64 string of float32 values in the server's native byte order.
	Embedding EmbeddingVector `json:"embedding"`
	// EncodingFormat is "base64" for a base64 vector and is absent for a float array.
	EncodingFormat *llama.EmbeddingEncodingFormat `json:"encoding_format,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingUsage counts the input tokens for the whole request.
type EmbeddingUsage struct {
	// PromptTokens counts the tokens evaluated across all inputs.
	PromptTokens int `json:"prompt_tokens"`
	// TotalTokens equals PromptTokens because embedding requests do not generate tokens.
	TotalTokens int `json:"total_tokens"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingVector is a numeric vector decoded from a float array or a base64 string.
type EmbeddingVector = llama.EmbeddingVector

// Embeddings sends a request to POST /v1/embeddings. The server must use pooled embeddings.
func (c *Client) Embeddings(ctx context.Context, req *EmbeddingRequest, opts ...llama.RequestOption) (*EmbeddingResponse, error) {
	if req == nil {
		return nil, errors.New("openai: nil embeddings request")
	}
	if req.Input.IsZero() {
		if _, legacy := req.Extra["content"]; legacy {
			return nil, errors.New("openai: use llama.Client.Embeddings for legacy content-only requests")
		}
	}
	return llama.Call[*EmbeddingResponse](ctx, c.c, http.MethodPost, embeddingsPath, req, opts...)
}
