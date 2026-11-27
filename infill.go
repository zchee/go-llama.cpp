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
	"errors"
	"fmt"
	"iter"
	"net/http"

	"github.com/zchee/go-llama.cpp/internal/streamerr"
)

// InfillRequest generates text between InputPrefix and InputSuffix. Each accepts text or a flat array
// of strings and token ids. The server appends Prompt to the input prefix, before the FIM middle token.
type InfillRequest struct {
	SamplingParams
	// InputPrefix is the required text or token sequence before the insertion point.
	InputPrefix InfillInput `json:"input_prefix"`
	// InputSuffix is the required text or token sequence after the insertion point.
	InputSuffix InfillInput `json:"input_suffix"`
	// InputExtra supplies optional file snippets before the fill-in-the-middle sequence.
	InputExtra []InfillContext `json:"input_extra,omitzero"`
	// Prompt appends optional text to InputPrefix, not after the middle marker.
	Prompt *string `json:"prompt,omitzero"`
	// Model selects the model to route to; required on a router and optional on a single-model server.
	Model string `json:"model,omitzero"`
	// NCmpl is the number of insertions to generate, from 1 to the server's parallel slot count; defaults to 1.
	NCmpl *int `json:"n_cmpl,omitzero"`
	// Stream is set by the typed client methods to choose server-sent events or buffered JSON.
	Stream *bool `json:"stream,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// InfillInput is a prefix or suffix. Unlike a completion prompt, a flat array of strings is a single input,
// not a batch. Set Text for a string, or Parts for an array of TextPart and TokenPart values.
type InfillInput struct {
	// Text is sent as a single JSON string when Parts is nil.
	Text string
	// Parts is sent as one sequence of text fragments and token IDs; leave Text empty when using it.
	Parts []PromptPart
}

// MarshalJSONTo implements json.MarshalerTo.
func (in InfillInput) MarshalJSONTo(enc *jsontext.Encoder) error {
	if in.Parts == nil {
		return enc.WriteToken(jsontext.String(in.Text))
	}
	if in.Text != "" {
		return errors.New("llama: infill input cannot specify both text and parts")
	}
	return writePromptParts(enc, in.Parts)
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (in *InfillInput) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var v InfillInput
	switch dec.PeekKind() {
	case '"':
		if err := json.UnmarshalDecode(dec, &v.Text); err != nil {
			return err
		}
	case '[':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		v.Parts = []PromptPart{}
		for dec.PeekKind() != ']' {
			var p PromptPart
			switch dec.PeekKind() {
			case '"':
				if err := json.UnmarshalDecode(dec, &p.text); err != nil {
					return err
				}
			case '0':
				p.isToken = true
				if err := json.UnmarshalDecode(dec, &p.token); err != nil {
					return err
				}
			default:
				return errors.New("llama: infill array elements must be strings or token ids")
			}
			v.Parts = append(v.Parts, p)
		}
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
	default:
		return errors.New("llama: infill input must be a string or a flat array of strings and token ids")
	}
	*in = v
	return nil
}

// InfillContext contributes a file's text as additional context for infilling.
type InfillContext struct {
	// Text is the required snippet to include in the model's fill-in-the-middle context.
	Text string `json:"text"`
	// Filename labels the snippet when the model supports file separators; defaults to "tmp".
	Filename *string `json:"filename,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Infill generates one fill-in-the-middle result. The model must support infill tokens.
func (c *Client) Infill(ctx context.Context, req *InfillRequest, opts ...RequestOption) (*CompletionResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil infill request")
	}
	results, err := c.InfillBatch(ctx, req, opts...)
	if err != nil {
		return nil, err
	}
	if len(results) != 1 {
		return nil, fmt.Errorf("llama: got %d infills, want one; use InfillBatch for multiple results", len(results))
	}
	return results[0], nil
}

// InfillBatch returns every infill generated for the request, including siblings requested with NCmpl.
func (c *Client) InfillBatch(ctx context.Context, req *InfillRequest, opts ...RequestOption) ([]*CompletionResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil infill batch request")
	}
	if req.Stream != nil && *req.Stream {
		cp := *req
		cp.Stream = new(false)
		req = &cp
	}
	return requireCompletionResults(Call[completionResults](ctx, c, http.MethodPost, "/infill", req, opts...))
}

// InfillStream streams fill-in-the-middle results without modifying req. Projected final frames have
// the same detection and indexing limits as CompletionStream.
func (c *Client) InfillStream(ctx context.Context, req *InfillRequest, opts ...RequestOption) iter.Seq2[*CompletionResponse, error] {
	if req == nil {
		return streamerr.Once[CompletionResponse](errors.New("llama: nil infill stream request"), ErrStreamConsumed)
	}
	cp := *req
	cp.Stream = new(true)
	return Stream[CompletionResponse](ctx, c, http.MethodPost, "/infill", &cp, opts...)
}
