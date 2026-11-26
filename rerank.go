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
	json "encoding/json/v2"
	"fmt"
	"net/http"
)

// RerankInput is a query or candidate document. Its zero value is an empty string.
type RerankInput struct {
	text   string
	object *RerankObject
}

// RerankObject is an input in object form. The server concatenates text fragments without a separator.
type RerankObject struct {
	// Text is a string or a list of fragments.
	Text StringOrSlice `json:"text,omitzero"`
	// Image is one image or a list, specified by URL, data URI, or raw base64.
	Image StringOrSlice             `json:"image,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// RerankText constructs a plain string query or document.
func RerankText(text string) RerankInput { return RerankInput{text: text} }

// RerankObjectInput constructs a query or document in object form.
func RerankObjectInput(o RerankObject) RerankInput { return RerankInput{object: &o} }

// Text returns the plain string and whether the input is in string form.
func (v RerankInput) Text() (string, bool) { return v.text, v.object == nil }

// Object returns the object, or nil for a plain string.
func (v RerankInput) Object() *RerankObject { return v.object }

// MarshalJSONTo encodes the input in the form in which it was built or decoded.
func (v RerankInput) MarshalJSONTo(enc *jsontext.Encoder) error {
	if v.object != nil {
		return json.MarshalEncode(enc, v.object)
	}
	return enc.WriteToken(jsontext.String(v.text))
}

// UnmarshalJSONFrom decodes a string or an object containing text and images.
func (v *RerankInput) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '"':
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		*v = RerankText(tok.String())
	case '{':
		var o RerankObject
		if err := json.UnmarshalDecode(dec, &o); err != nil {
			return err
		}
		*v = RerankObjectInput(o)
	default:
		return fmt.Errorf("llama: rerank input must be a string or object, got %s", kind)
	}
	return nil
}

// RerankRequest ranks Documents in Jina format.
type RerankRequest struct {
	// Query is compared with every document.
	Query RerankInput `json:"query"`
	// Documents must contain at least one candidate.
	Documents []RerankInput `json:"documents"`
	// Model selects the model in router mode and is echoed in the response.
	Model *string `json:"model,omitzero"`
	// TopN limits the returned results; nil returns all, zero returns none.
	TopN  *int                      `json:"top_n,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// RerankResponse contains the Jina results in descending score order.
type RerankResponse struct {
	// Model identifies the model named in the request or the server's default model.
	Model string `json:"model"`
	// Object is "list".
	Object string `json:"object"`
	// Usage counts all scored pairs, including those excluded by TopN.
	Usage RerankUsage `json:"usage"`
	// Results are the selected documents and their scores.
	Results []RerankResult            `json:"results"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// RerankUsage counts the tokens evaluated for all query-document pairs.
type RerankUsage struct {
	// PromptTokens is the sum of the pair lengths in tokens.
	PromptTokens int `json:"prompt_tokens"`
	// TotalTokens equals PromptTokens because reranking does not generate tokens.
	TotalTokens int                       `json:"total_tokens"`
	Extra       map[string]jsontext.Value `json:",embed"`
}

// RerankResult is one Jina result.
type RerankResult struct {
	// Index is the zero-based document index in the request.
	Index int `json:"index"`
	// RelevanceScore is the model's score for the document.
	RelevanceScore float64                   `json:"relevance_score"`
	Extra          map[string]jsontext.Value `json:",embed"`
}

// RerankTEIRequest ranks Texts in Text Embeddings Inference format.
type RerankTEIRequest struct {
	// Query is compared with every candidate.
	Query RerankInput `json:"query"`
	// Texts must contain at least one candidate.
	Texts []RerankInput `json:"texts"`
	// Model selects the model in router mode.
	Model *string `json:"model,omitzero"`
	// TopN limits the returned results; nil returns all, zero returns none.
	TopN *int `json:"top_n,omitzero"`
	// ReturnText asks the server to echo the text of each returned document.
	ReturnText *bool                     `json:"return_text,omitzero"`
	Extra      map[string]jsontext.Value `json:",embed"`
}

// RerankTEIResult is one entry of the bare TEI result array.
type RerankTEIResult struct {
	// Index is the zero-based document index in the request.
	Index int `json:"index"`
	// Score is the model's score for the document.
	Score float64 `json:"score"`
	// Text holds the concatenated document text only when ReturnText is true.
	Text  *string                   `json:"text,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// Rerank scores documents with POST /reranking and returns the wrapped Jina response.
func (c *Client) Rerank(ctx context.Context, req *RerankRequest, opts ...RequestOption) (*RerankResponse, error) {
	return Call[*RerankResponse](ctx, c, http.MethodPost, "/reranking", req, opts...)
}

// RerankTEI scores documents with POST /reranking and returns TEI results in descending score order.
func (c *Client) RerankTEI(ctx context.Context, req *RerankTEIRequest, opts ...RequestOption) ([]RerankTEIResult, error) {
	return Call[[]RerankTEIResult](ctx, c, http.MethodPost, "/reranking", req, opts...)
}
