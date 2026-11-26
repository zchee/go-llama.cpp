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
	"fmt"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
)

var _ = registerContracts(map[string]contractCase{
	"POST /reranking": {
		server: rerankPreset,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.Rerank(t.Context(), &llama.RerankRequest{
				Query:     llama.RerankText("capital of France"),
				Documents: []llama.RerankInput{llama.RerankText("Paris")},
			})
			if err != nil {
				return err
			}
			if res == nil || res.Model != "jina-reranker" || res.Object != "list" || len(res.Results) != 1 || res.Results[0].Index != 0 || res.Usage.PromptTokens <= 0 || res.Usage.TotalTokens != res.Usage.PromptTokens {
				return fmt.Errorf("unexpected Jina ranking: %+v", res)
			}
			return nil
		},
	},
})
