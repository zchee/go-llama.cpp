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
	"POST /v1/chat/completions": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		res, err := openai.NewClient(c).ChatCompletions(t.Context(), chatRequest())
		if err != nil {
			return err
		}
		if res.Object != "chat.completion" || len(res.Choices) != 1 || res.Choices[0].Message.Role != "assistant" {
			return fmt.Errorf("unexpected chat completion: %+v", res)
		}
		return nil
	}},
	"POST /v1/chat/completions/control": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		res, err := openai.NewClient(c).ControlChatCompletion(t.Context(), &openai.ControlChatCompletionRequest{ID: "inactive", Action: "reasoning_end"})
		if err != nil {
			return err
		}
		if res.Success || res.Message == nil || *res.Message != "no active completion for this id" {
			return fmt.Errorf("unexpected inactive completion status: %+v", res)
		}
		return nil
	}},
	"POST /v1/chat/completions/input_tokens": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		res, err := openai.NewClient(c).ChatInputTokens(t.Context(), chatRequest())
		if err != nil {
			return err
		}
		if res.Object != "response.input_tokens" || res.InputTokens <= 0 {
			return fmt.Errorf("unexpected chat token count: %+v", res)
		}
		return nil
	}},
})
