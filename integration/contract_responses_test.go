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

var _ = registerContracts(map[string]contractCase{
	"POST /v1/responses": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		res, err := openai.NewClient(c).Responses(t.Context(), &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("Once upon a time")}, MaxOutputTokens: new(4)})
		if err != nil {
			return err
		}
		if res == nil || res.Status != "completed" || res.Usage == nil || res.Usage.OutputTokens <= 0 {
			return fmt.Errorf("expected completed response with token usage, got %+v", res)
		}
		return nil
	}},
	"POST /v1/responses/input_tokens": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		res, err := openai.NewClient(c).ResponsesInputTokens(t.Context(), &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("Once upon a time")}})
		if err != nil {
			return err
		}
		if res == nil || res.Object != "response.input_tokens" || res.InputTokens <= 0 {
			return fmt.Errorf("expected a positive input token count, got %+v", res)
		}
		return nil
	}},
})
