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

package integration

import (
	"fmt"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

var _ = registerContracts(map[string]contractCase{
	"POST /embedding": {
		server: presetEmbeddingPooled,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.Embedding(t.Context(), &llama.EmbeddingRequest{Content: llama.EmbeddingText("hello")})
			if err != nil {
				return err
			}
			if len(res) != 1 || res[0].Index != 0 || len(res[0].Embedding) != 1 || res[0].Embedding[0].Len() != 384 {
				return fmt.Errorf("legacy embedding: want one result with one 384-dimensional vector, got %d results", len(res))
			}
			return nil
		},
	},
	"POST /embeddings": {
		server: presetEmbeddingPerToken,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.Embeddings(t.Context(), &llama.EmbeddingRequest{Input: llama.EmbeddingTokens(101, 7592, 102)})
			if err != nil {
				return err
			}
			if len(res) != 1 || res[0].Index != 0 || len(res[0].Embedding) != 3 {
				return fmt.Errorf("per-token embedding: want one result with three vectors, got %d results", len(res))
			}
			for i, vector := range res[0].Embedding {
				if vector.Len() != 384 {
					return fmt.Errorf("embedding token %d: got %d dimensions, want 384", i, vector.Len())
				}
			}
			return nil
		},
	},
	"POST /v1/embeddings": {
		server: presetEmbeddingPooled,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := openai.NewClient(c).Embeddings(t.Context(), &openai.EmbeddingRequest{
				Input:          llama.EmbeddingTexts("hello", "world"),
				EncodingFormat: new(llama.EmbeddingEncodingBase64),
			})
			if err != nil {
				return err
			}
			if res.Object != "list" || res.Model != "bert-bge-small" || len(res.Data) != 2 || res.Usage.PromptTokens != 6 || res.Usage.TotalTokens != 6 {
				return fmt.Errorf("OpenAI embedding: object=%q model=%q results=%d usage=%+v", res.Object, res.Model, len(res.Data), res.Usage)
			}
			for i, item := range res.Data {
				if item.Index != i || item.Embedding.Len() != 384 || item.Embedding.Format() != llama.EmbeddingEncodingBase64 {
					return fmt.Errorf("OpenAI embedding result %d: index=%d dimensions=%d format=%q", i, item.Index, item.Embedding.Len(), item.Embedding.Format())
				}
			}
			return nil
		},
	},
})
