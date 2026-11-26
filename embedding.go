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

// EmbeddingEncodingFormat selects the wire representation of an OpenAI-compatible vector.
type EmbeddingEncodingFormat string

const (
	// EmbeddingEncodingFloat uses a JSON array of numbers.
	EmbeddingEncodingFloat EmbeddingEncodingFormat = "float"
	// EmbeddingEncodingBase64 uses the server's native float32 bytes. The client assumes a
	// little-endian server; use EmbeddingEncodingFloat if the server's byte order is unknown.
	EmbeddingEncodingBase64 EmbeddingEncodingFormat = "base64"
)

const (
	embeddingPath  = "/embedding"
	embeddingsPath = "/embeddings"
)

// EmbeddingRequest is the request shared by the two native embedding endpoints.
type EmbeddingRequest struct {
	// Input supplies one input or a batch to embed and takes precedence over Content.
	Input EmbeddingInput `json:"input,omitzero"`
	// Content supplies the inputs when Input is absent; the server requires at least one of the two members.
	Content EmbeddingInput `json:"content,omitzero"`
	// Model is required in router mode to select a model and is ignored in single-model mode.
	Model string `json:"model,omitzero"`
	// Normalize selects -1 (none), 0 (max absolute), 1 (L1), 2 (L2), or a larger p (p-norm) for pooled vectors,
	// using the server setting when omitted and having no effect with pooling none.
	Normalize *int `json:"embd_normalize,omitzero"`
	// EncodingFormat accepts "float" or "base64", but the native response always uses numeric arrays.
	EncodingFormat *EmbeddingEncodingFormat `json:"encoding_format,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingResponse has one result per input.
type EmbeddingResponse []EmbeddingResult

// EmbeddingResult contains the vectors for one input.
type EmbeddingResult struct {
	// Index is the zero-based position of this input in the request batch.
	Index int `json:"index"`
	// Embedding is a list containing one pooled vector, or one vector per token with pooling none.
	Embedding []EmbeddingVector `json:"embedding"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Embedding sends a request to the legacy native POST /embedding endpoint, an alias of Embeddings.
func (c *Client) Embedding(ctx context.Context, req *EmbeddingRequest, opts ...RequestOption) (EmbeddingResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil embedding request")
	}
	return Call[EmbeddingResponse](ctx, c, http.MethodPost, embeddingPath, req, opts...)
}

// Embeddings sends a request to POST /embeddings, including on servers configured with pooling none.
func (c *Client) Embeddings(ctx context.Context, req *EmbeddingRequest, opts ...RequestOption) (EmbeddingResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil embeddings request")
	}
	return Call[EmbeddingResponse](ctx, c, http.MethodPost, embeddingsPath, req, opts...)
}
