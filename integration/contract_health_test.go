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

// The contract cases of GET /health. Copy this file to add the cases of other endpoints: name it
// contract_<name>_test.go, register its cases with registerContracts, and call only typed methods.
var _ = registerContracts(map[string]contractCase{
	"GET /health": {
		server: presetStories260K,
		call: func(t *testing.T, c *llama.Client) error {
			res, err := c.Health(t.Context())
			if err != nil {
				return err
			}
			if res.Status != "ok" {
				return fmt.Errorf("status = %q, want %q", res.Status, "ok")
			}
			return nil
		},
	},
})
