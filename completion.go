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

// CompletionRequest selects the prompt and sampling settings of a native text completion.
type CompletionRequest struct {
	SamplingParams
	// Prompt supplies the text, token IDs, or multimodal input to complete; a batch supplies several prompts.
	Prompt Prompt `json:"prompt"`
	// Model selects the model to route to; required on a router and optional on a single-model server.
	Model string `json:"model,omitzero"`
	// NCmpl is the number of completions per prompt, from 1 to the server's parallel slot count; defaults to 1.
	NCmpl *int `json:"n_cmpl,omitzero"`
	// Stream is set by the typed client methods to choose server-sent events or buffered JSON.
	Stream *bool `json:"stream,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// CompletionResponse is a final result, token event or prompt progress event from /completion or /infill.
// Every field can be absent: response_fields projects the final object onto the selected keys. A nested
// selection such as "generation_settings/n_predict" is returned under that literal key in Extra.
type CompletionResponse struct {
	// Index is the zero-based completion position in the batch, also used to associate streamed events.
	Index *int `json:"index,omitzero"`
	// Content is the full buffered text or the current stream fragment, excluding any stopping word.
	Content *string `json:"content,omitzero"`
	// Tokens lists generated vocabulary IDs; populated for ReturnTokens or a partial stream event.
	Tokens []Token `json:"tokens,omitzero"`
	// IDSlot identifies the server slot that processed this completion.
	IDSlot *int `json:"id_slot,omitzero"`
	// Stop is true in the final result and false in partial stream events.
	Stop *bool `json:"stop,omitzero"`
	// Model is the model alias reported in the final result.
	Model *string `json:"model,omitzero"`
	// TokensPredicted counts tokens generated so far, not just those in the current event.
	TokensPredicted *int `json:"tokens_predicted,omitzero"`
	// TokensEvaluated counts input tokens, including any reused from the prompt cache.
	TokensEvaluated *int `json:"tokens_evaluated,omitzero"`
	// GenerationSettings reports the effective parameters in the final result, after the server applies defaults.
	GenerationSettings *GenerationSettings `json:"generation_settings,omitzero"`
	// Prompt is the detokenized model input in the final result, including any added special tokens.
	Prompt *string `json:"prompt,omitzero"`
	// HasNewLine reports in the final result whether generation reached a newline.
	HasNewLine *bool `json:"has_new_line,omitzero"`
	// Truncated reports in the final result whether the context limit caused tokens to be dropped.
	Truncated *bool `json:"truncated,omitzero"`
	// StopType states why generation ended, for example "eos", "limit", or "word"; absent in partial events.
	StopType *string `json:"stop_type,omitzero"`
	// StoppingWord is the stop string matched in the final result, or empty if no stop string matched.
	StoppingWord *string `json:"stopping_word,omitzero"`
	// TokensCached counts tokens retained in the slot at the end of generation, not just prompt cache hits.
	TokensCached *int `json:"tokens_cached,omitzero"`
	// Timings reports prompt and generation speeds in the final result and timed partial events.
	Timings *Timings `json:"timings,omitzero"`
	// PromptProgress is present in prompt processing events when ReturnProgress is enabled.
	PromptProgress *PromptProgress `json:"prompt_progress,omitzero"`
	// CompletionProbabilities is populated when NProbs is positive, on buffered results or token events.
	CompletionProbabilities []TokenProb `json:"completion_probabilities,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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

// Completion generates exactly one native text completion. Use CompletionBatch for requests returning several results.
func (c *Client) Completion(ctx context.Context, req *CompletionRequest, opts ...RequestOption) (*CompletionResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil completion request")
	}
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
		return nil, errors.New("llama: nil completion batch request")
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
		return streamerr.Once[CompletionResponse](errors.New("llama: nil completion stream request"), ErrStreamConsumed)
	}
	cp := *req
	cp.Stream = new(true)
	return Stream[CompletionResponse](ctx, c, http.MethodPost, "/completion", &cp, opts...)
}
