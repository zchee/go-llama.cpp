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
	"github.com/zchee/go-llama.cpp/openai"
)

// The contract case of POST /v1/completions.
var _ = registerContracts(map[string]contractCase{
	"POST /v1/completions": {
		server: presetStories260K,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := openai.NewClient(c).Completions(t.Context(), &openai.CompletionRequest{
				Model:       "tinyllama-2",
				Prompt:      llama.TextPrompt("Once upon a time"),
				MaxTokens:   new(8),
				Temperature: new(0.0),
			})
			if err != nil {
				return err
			}
			if res.Object != "text_completion" {
				return fmt.Errorf("object = %q, want %q", res.Object, "text_completion")
			}
			if len(res.Choices) != 1 || res.Choices[0].Text == "" {
				return fmt.Errorf("choices = %+v, want 1 non-empty choice", res.Choices)
			}
			if res.Usage == nil || res.Usage.CompletionTokens == 0 {
				return fmt.Errorf("usage = %+v, want populated usage", res.Usage)
			}
			return nil
		},
	},
})
