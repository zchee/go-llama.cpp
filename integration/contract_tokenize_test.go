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
	"POST /tokenize": {
		server: presetStories260K,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.Tokenize(t.Context(), &llama.TokenizeRequest{Content: llama.TokenizeText("Hello world!")})
			if err != nil {
				return err
			}
			if res == nil || len(res.Tokens) == 0 {
				return fmt.Errorf("expected token IDs, got %+v", res)
			}
			return nil
		},
	},
	"POST /detokenize": {
		server: presetStories260K,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.Detokenize(t.Context(), &llama.DetokenizeRequest{Tokens: []llama.Token{198, 164}})
			if err != nil {
				return err
			}
			if res == nil || res.Content != "á" {
				return fmt.Errorf("expected content %q, got %+v", "á", res)
			}
			return nil
		},
	},
	"POST /apply-template": {
		server: presetStories260K,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.ApplyTemplate(t.Context(), &llama.ApplyTemplateRequest{Messages: []llama.Message{
				{Role: "user", Content: llama.TextContent("Hello")},
			}})
			if err != nil {
				return err
			}
			if res == nil || res.Prompt == "" {
				return fmt.Errorf("expected a rendered prompt, got %+v", res)
			}
			return nil
		},
	},
})
