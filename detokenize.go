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

// DetokenizeRequest is the body sent to POST /detokenize.
type DetokenizeRequest struct {
	// Tokens are the token IDs to convert back to text.
	Tokens []Token `json:"tokens,omitzero"`

	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// DetokenizeResponse is the JSON returned by POST /detokenize.
type DetokenizeResponse struct {
	// Content is the text spelled by the request tokens.
	Content string `json:"content"`

	// Extra preserves unrecognized JSON members.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Detokenize converts token IDs to text using the model's vocabulary.
func (c *Client) Detokenize(ctx context.Context, req *DetokenizeRequest, opts ...RequestOption) (*DetokenizeResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil detokenize request")
	}
	return Call[*DetokenizeResponse](ctx, c, http.MethodPost, "/detokenize", req, opts...)
}
