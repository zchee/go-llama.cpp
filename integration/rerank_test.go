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
	"errors"
	"net/http"
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
		topN  *int
		echo  bool
		count int
	}{
		"success: all candidates with text":             {echo: true, count: 2},
		"success: highest score only without text":      {topN: new(1), count: 1},
		"success: explicit zero returns an empty array": {topN: new(0)},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := c.RerankTEI(t.Context(), &llama.RerankTEIRequest{Query: llama.RerankText("Which city is the capital of France?"), Texts: docs, TopN: tt.topN, ReturnText: new(tt.echo)})
			if err != nil {
				t.Fatal(err)
			}
			if got == nil || len(got) != tt.count {
				t.Fatalf("results = %+v, want an array with %d entries", got, tt.count)
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

func TestRerankErrors(t *testing.T) {
	doc := llama.RerankText("Paris")
	tests := map[string]struct {
		preset *serverPreset
		req    llama.RerankRequest
		status int
		kind   llama.ErrorType
	}{
		"error: text model cannot accept images": {rerankPreset, llama.RerankRequest{Query: llama.RerankObjectInput(llama.RerankObject{Image: llama.StringValue("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/k1sAAAAASUVORK5CYII=")}), Documents: []llama.RerankInput{doc}}, http.StatusBadRequest, llama.ErrorTypeInvalidRequest},
		"error: no documents":                    {rerankPreset, llama.RerankRequest{Query: doc}, http.StatusBadRequest, llama.ErrorTypeInvalidRequest},
		"error: negative result limit":           {rerankPreset, llama.RerankRequest{Query: doc, Documents: []llama.RerankInput{doc}, TopN: new(-1)}, http.StatusBadRequest, llama.ErrorTypeInvalidRequest},
		"error: reranking disabled":              {presetStories260K, llama.RerankRequest{Query: doc, Documents: []llama.RerankInput{doc}}, http.StatusNotImplemented, llama.ErrorTypeNotSupported},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := tt.preset.server(t).Client(t)
			_, err := c.Rerank(t.Context(), &tt.req)
			if apiErr, ok := errors.AsType[*llama.APIError](err); !ok || apiErr.StatusCode != tt.status || apiErr.Type != tt.kind {
				t.Errorf("Rerank error = %v, want %d %s", err, tt.status, tt.kind)
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
			req := llama.RerankRequest{Query: llama.RerankText("city"), Documents: []llama.RerankInput{llama.RerankText("Paris")}, Model: new("alias-test")}
			got, err := llama.Call[llama.RerankResponse](t.Context(), c, http.MethodPost, tt.path, &req)
			if err != nil {
				t.Fatal(err)
			}
			if got.Model != *req.Model || len(got.Results) != 1 || got.Results[0].Index != 0 {
				t.Fatalf("POST %s: unexpected result: %+v", tt.path, got)
			}
		})
	}
}
