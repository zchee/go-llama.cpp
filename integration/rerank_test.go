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

package integration

import (
	"encoding/json/jsontext"
	"errors"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

var rerankPreset = &serverPreset{
	name: "jina-reranker-tiny",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "jina-reranker-tiny"),
			Args:  []string{"--reranking", "--ctx-size", "512", "--parallel", "1", "--alias", "jina-reranker"},
		}
	},
}

func TestRerankTEI(t *testing.T) {
	c, _ := rerankPreset.server(t).Client(t)
	docs := []llama.RerankInput{llama.RerankObjectInput(llama.RerankObject{Text: llama.StringSlice("Apples are ", "a fruit.")}), llama.RerankText("Paris is the capital of France.")}
	texts := []string{"Apples are a fruit.", "Paris is the capital of France."}
	tests := map[string]struct {
		topN  *int32
		echo  bool
		count int
	}{
		"success: all candidates with text":             {echo: true, count: 2},
		"success: highest score only without text":      {topN: new(int32(1)), count: 1},
		"success: explicit zero returns an empty array": {topN: new(int32(0))},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req := llama.RerankTEIRequest{Query: llama.RerankText("Which city is the capital of France?"), Texts: docs, TopN: tt.topN, ReturnText: new(tt.echo)}
			got, err := c.RerankTEI(t.Context(), &req)
			if err != nil {
				t.Fatal(err)
			}
			if got == nil || len(got) != tt.count {
				t.Fatalf("results = %+v, want an array with %d entries", got, tt.count)
			}
			if tt.topN != nil && *tt.topN == 1 {
				req.TopN = nil
				full, err := c.RerankTEI(t.Context(), &req)
				if err != nil {
					t.Fatal(err)
				}
				if len(full) != len(docs) {
					t.Fatalf("full ranking has %d results; want %d", len(full), len(docs))
				}
				if got[0].Index != full[0].Index {
					t.Errorf("top_n: 1 returned index %d; full ranking starts with index %d", got[0].Index, full[0].Index)
				}
			}
			seen := make(map[int]bool)
			for i, result := range got {
				if result.Index < 0 || result.Index >= len(docs) || seen[result.Index] {
					t.Fatalf("unexpected document index %d in %+v", result.Index, got)
				}
				seen[result.Index] = true
				if i > 0 && result.Score > got[i-1].Score {
					t.Errorf("results not in descending score order: %+v", got)
				}
				if tt.echo != (result.Text != nil) {
					t.Fatalf("document %d: text present = %t, want %t", result.Index, result.Text != nil, tt.echo)
				}
				if result.Text != nil && *result.Text != texts[result.Index] {
					t.Errorf("document %d text = %q, want %q", result.Index, *result.Text, texts[result.Index])
				}
			}
		})
	}
}

func TestRerankCandidateLists(t *testing.T) {
	srv := rerankPreset.server(t)
	query := llama.RerankText("Paris")
	docs := []llama.RerankInput{query}
	tests := map[string]struct {
		candidates []llama.RerankInput
		extra      jsontext.Value
	}{
		"error: nil candidates":                           {},
		"error: empty candidates":                         {candidates: []llama.RerankInput{}},
		"error: other format's candidate key":             {candidates: docs, extra: jsontext.Value(`["bread","milk"]`)},
		"error: other format's candidate key set to null": {candidates: docs, extra: jsontext.Value(`null`)},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, observer := srv.Client(t)
			var jinaExtra, teiExtra map[string]jsontext.Value
			if tt.extra != nil {
				jinaExtra = map[string]jsontext.Value{"texts": tt.extra}
				teiExtra = map[string]jsontext.Value{"documents": tt.extra}
			}
			_, jinaErr := c.Rerank(t.Context(), &llama.RerankRequest{Query: query, Documents: tt.candidates, Extra: jinaExtra})
			_, teiErr := c.RerankTEI(t.Context(), &llama.RerankTEIRequest{Query: query, Texts: tt.candidates, Extra: teiExtra})
			if tt.extra != nil {
				if jinaErr == nil || !strings.Contains(jinaErr.Error(), `"texts"`) {
					t.Errorf(`Rerank error = %v, want rejection of "texts"`, jinaErr)
				}
				if teiErr == nil || !strings.Contains(teiErr.Error(), `"documents"`) {
					t.Errorf(`RerankTEI error = %v, want rejection of "documents"`, teiErr)
				}
				if got := len(observer.Exchanges()); got != 0 {
					t.Errorf("sent %d requests despite conflicting candidate keys, want none", got)
				}
				return
			}
			for _, err := range []error{jinaErr, teiErr} {
				if apiErr, ok := errors.AsType[*llama.APIError](err); !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest {
					t.Errorf("reranking error = %v, want HTTP 400 invalid_request_error", err)
				}
			}
		})
	}
}

func TestRerankRejectsExtraTopN(t *testing.T) {
	srv := rerankPreset.server(t)
	query := llama.RerankText("France")
	docs := []llama.RerankInput{llama.RerankText("Paris")}
	extra := map[string]jsontext.Value{"top_n": jsontext.Value(`4294967296`)}
	tests := map[string]struct {
		call func(*testing.T, *llama.Client) error
	}{
		"error: Jina top_n in Extra": {call: func(t *testing.T, c *llama.Client) error {
			_, err := c.Rerank(t.Context(), &llama.RerankRequest{Query: query, Documents: docs, Extra: extra})
			return err
		}},
		"error: TEI top_n in Extra": {call: func(t *testing.T, c *llama.Client) error {
			_, err := c.RerankTEI(t.Context(), &llama.RerankTEIRequest{Query: query, Texts: docs, Extra: extra})
			return err
		}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, observer := srv.Client(t)
			if err := tt.call(t, c); err == nil || !strings.Contains(err.Error(), `"top_n"`) {
				t.Errorf("reranking error = %v, want a conflict for top_n", err)
			}
			if got := len(observer.Exchanges()); got != 0 {
				t.Errorf("sent %d requests despite conflicting top_n, want none", got)
			}
		})
	}
}

func TestRerankErrors(t *testing.T) {
	doc := llama.RerankText("Paris")
	tests := map[string]struct {
		preset  *serverPreset
		req     llama.RerankRequest
		status  int
		kind    llama.ErrorType
		message string
	}{
		"error: text model cannot accept images": {rerankPreset, llama.RerankRequest{Query: llama.RerankObjectInput(llama.RerankObject{Image: llama.StringValue("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/k1sAAAAASUVORK5CYII=")}), Documents: []llama.RerankInput{doc}}, http.StatusBadRequest, llama.ErrorTypeInvalidRequest, "multimodal rerank input requires --mmproj"},
		"error: negative result limit":           {rerankPreset, llama.RerankRequest{Query: doc, Documents: []llama.RerankInput{doc}, TopN: new(int32(-1))}, http.StatusBadRequest, llama.ErrorTypeInvalidRequest, ""},
		"error: reranking disabled":              {presetStories260K, llama.RerankRequest{Query: doc, Documents: []llama.RerankInput{doc}}, http.StatusNotImplemented, llama.ErrorTypeNotSupported, ""},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := tt.preset.server(t).Client(t)
			_, err := c.Rerank(t.Context(), &tt.req)
			apiErr, ok := errors.AsType[*llama.APIError](err)
			if !ok || apiErr.StatusCode != tt.status || apiErr.Type != tt.kind {
				t.Fatalf("Rerank error = %v, want %d %s", err, tt.status, tt.kind)
			}
			if tt.message != "" && apiErr.Message != tt.message {
				t.Errorf("Rerank error message = %q, want %q", apiErr.Message, tt.message)
			}
		})
	}
}

func TestRerankAliases(t *testing.T) {
	c, _ := rerankPreset.server(t).Client(t)
	tests := map[string]struct{ path string }{
		"success: rerank":              {"/rerank"},
		"success: versioned rerank":    {"/v1/rerank"},
		"success: versioned reranking": {"/v1/reranking"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req := llama.RerankRequest{Query: llama.RerankText("city"), Documents: []llama.RerankInput{llama.RerankText("Paris")}, Model: "alias-test"}
			got, err := llama.Call[llama.RerankResponse](t.Context(), c, http.MethodPost, tt.path, &req)
			if err != nil {
				t.Fatal(err)
			}
			if got.Model != req.Model || len(got.Results) != 1 || got.Results[0].Index != 0 {
				t.Fatalf("POST %s: unexpected result: %+v", tt.path, got)
			}
		})
	}
}
