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
)

// CompletionRequest selects the prompt and sampling settings of a native text completion.
type CompletionRequest struct {
	SamplingParams
	Prompt Prompt `json:"prompt"`
	// Model selects a model when the request goes through a router.
	Model string `json:"model,omitzero"`
	// NCmpl is the number of completions to generate for each prompt.
	NCmpl *int `json:"n_cmpl,omitzero"`
	// Stream is selected automatically by the typed client methods.
	Stream *bool                     `json:"stream,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// CompletionResponse is a final result, token event or prompt progress event from /completion or /infill.
// Every field can be absent: response_fields projects the final object onto the selected keys. A nested
// selection such as "generation_settings/n_predict" is returned under that literal key in Extra.
type CompletionResponse struct {
	Index                   *int                      `json:"index,omitzero"`
	Content                 *string                   `json:"content,omitzero"`
	Tokens                  []Token                   `json:"tokens,omitzero"`
	IDSlot                  *int                      `json:"id_slot,omitzero"`
	Stop                    *bool                     `json:"stop,omitzero"`
	Model                   *string                   `json:"model,omitzero"`
	TokensPredicted         *int                      `json:"tokens_predicted,omitzero"`
	TokensEvaluated         *int                      `json:"tokens_evaluated,omitzero"`
	GenerationSettings      *GenerationSettings       `json:"generation_settings,omitzero"`
	Prompt                  *string                   `json:"prompt,omitzero"`
	HasNewLine              *bool                     `json:"has_new_line,omitzero"`
	Truncated               *bool                     `json:"truncated,omitzero"`
	StopType                *string                   `json:"stop_type,omitzero"`
	StoppingWord            *string                   `json:"stopping_word,omitzero"`
	TokensCached            *int                      `json:"tokens_cached,omitzero"`
	Timings                 *Timings                  `json:"timings,omitzero"`
	PromptProgress          *PromptProgress           `json:"prompt_progress,omitzero"`
	CompletionProbabilities []TokenProb               `json:"completion_probabilities,omitzero"`
	Extra                   map[string]jsontext.Value `json:",embed"`
}

// completionResults normalizes the server's single object or array into a slice at the transport boundary.
type completionResults []*CompletionResponse

func (r *completionResults) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch dec.PeekKind() {
	case '{':
		var result CompletionResponse
		if err := json.UnmarshalDecode(dec, &result); err != nil {
			return err
		}
		*r = completionResults{&result}
	case '[':
		var results []*CompletionResponse
		if err := json.UnmarshalDecode(dec, &results); err != nil {
			return err
		}
		// The server asserts at least one result for a valid request (GGML_ASSERT(!arr.empty())).
		if len(results) == 0 {
			return errors.New("llama: empty completion result array")
		}
		for _, result := range results {
			if result == nil {
				return errors.New("llama: null completion in result array")
			}
		}
		*r = results
	default:
		return errors.New("llama: expected a completion object or an array of completions")
	}
	return nil
}

// nativeStreamRequest supplies the lower bound even if a connection drops before another result starts.
func nativeStreamRequest(body any, request *http.Request) (int, []string) {
	var fields []string
	var prompts, count int
	var n *int
	var extra map[string]jsontext.Value
	switch r := body.(type) {
	case *CompletionRequest:
		if r != nil {
			prompts, n, extra, fields = len(r.Prompt.Batch()), r.NCmpl, r.Extra, r.ResponseFields
		}
	case *InfillRequest:
		if r != nil {
			n, extra, fields = r.NCmpl, r.Extra, r.ResponseFields
		}
	default:
		// For raw Stream calls read the already encoded body; do not invoke a user's marshaler twice.
		if request != nil && request.GetBody != nil {
			rd, err := request.GetBody()
			if err == nil {
				defer rd.Close()
				var params struct {
					Prompt         Prompt   `json:"prompt"`
					N              int      `json:"n"`
					NCmpl          *int     `json:"n_cmpl"`
					ResponseFields []string `json:"response_fields"`
				}
				if json.UnmarshalRead(rd, &params) == nil {
					prompts, n, count, fields = len(params.Prompt.Batch()), params.NCmpl, params.N, params.ResponseFields
				}
			}
		}
	}
	if n != nil {
		count = *n
	} else if alias := extra["n"]; alias != nil {
		_ = json.Unmarshal(alias, &count)
	}
	return max(1, prompts) * max(1, count), fields
}

var errNilCompletion = errors.New("llama: nil completion request")

// Completion generates exactly one native text completion. Use CompletionBatch for requests returning several results.
func (c *Client) Completion(ctx context.Context, req *CompletionRequest, opts ...RequestOption) (*CompletionResponse, error) {
	results, err := c.CompletionBatch(ctx, req, opts...)
	if err != nil {
		return nil, err
	}
	if len(results) != 1 {
		return nil, fmt.Errorf("llama: got %d completions, want one; use CompletionBatch for multiple results", len(results))
	}
	return results[0], nil
}

// CompletionBatch returns all completions, including a single completion as a one-element slice.
func (c *Client) CompletionBatch(ctx context.Context, req *CompletionRequest, opts ...RequestOption) ([]*CompletionResponse, error) {
	if req == nil {
		return nil, errNilCompletion
	}
	if req.Stream != nil && *req.Stream {
		cp := *req
		cp.Stream = new(false)
		req = &cp
	}
	return requireCompletionResults(Call[completionResults](ctx, c, http.MethodPost, "/completion", req, opts...))
}

func requireCompletionResults(results completionResults, err error) ([]*CompletionResponse, error) {
	if err == nil && len(results) == 0 {
		return nil, errors.New("llama: expected at least one completion result")
	}
	return results, err
}

// CompletionStream streams progress and token events followed by a final result for each prompt and completion.
// A stream that ends by itself before a final frame for each of the len(prompts)*NCmpl results, or whose last
// frame is not final, yields an error matching ErrIncompleteStream; a caller's break or a canceled ctx does not.
//
// The server projects response_fields onto final frames only; partial frames always carry stop:false,
// so a final frame whose projection removed stop is still recognized by the absence of stop. When the
// projection removes index, final frames are counted but cannot be attributed to individual results.
// A frame lost in the middle of a result cannot be detected if every expected final frame still arrives.
func (c *Client) CompletionStream(ctx context.Context, req *CompletionRequest, opts ...RequestOption) iter.Seq2[*CompletionResponse, error] {
	if req == nil {
		return singleUseStream(func(yield func(*CompletionResponse, error) bool) { yield(nil, errNilCompletion) })
	}
	cp := *req
	cp.Stream = new(true)
	return Stream[CompletionResponse](ctx, c, http.MethodPost, "/completion", &cp, opts...)
}
