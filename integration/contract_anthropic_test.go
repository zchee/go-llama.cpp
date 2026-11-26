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
	"github.com/zchee/go-llama.cpp/anthropic"
)

var _ = registerContracts(map[string]contractCase{
	"POST /v1/messages": {
		server:        presetAnthropic,
		clientOptions: anthropicClientOptions,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := anthropic.NewClient(c).Messages(t.Context(), anthropicRequest())
			if err != nil {
				return err
			}
			if res == nil || res.ID == "" || res.Type != "message" || res.Role != "assistant" || res.Usage.OutputTokens <= 0 || res.StopReason == nil {
				return fmt.Errorf("incomplete assistant reply: %+v", res)
			}
			return nil
		},
	},
	"POST /v1/messages/count_tokens": {
		server:        presetAnthropic,
		clientOptions: anthropicClientOptions,
		call: func(t *testing.T, c *llama.Client) error {
			req := anthropicRequest()
			req.MaxTokens = nil
			res, err := anthropic.NewClient(c).CountTokens(t.Context(), req)
			if err != nil {
				return err
			}
			if res == nil || res.InputTokens <= 0 {
				return fmt.Errorf("unexpected count: %+v", res)
			}
			return nil
		},
	},
})
