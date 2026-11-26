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
	"github.com/zchee/go-llama.cpp/systemone"
)

var _ = registerContracts(map[string]contractCase{
	"POST /v1/systemone": {server: presetSystemone, call: func(t *testing.T, c *llama.Client) error {
		res, err := systemone.NewClient(c).Evaluate(t.Context(), &systemone.Request{State: systemone.Text("The parcel has arrived."), Questions: map[string]systemone.Question{"arrived": systemone.Noul(systemone.Text("Was the parcel delivered?"), nil)}})
		if err != nil {
			return err
		}
		a, ok := res.Answers["arrived"].Value.(systemone.NoulAnswer)
		if !ok || a.Noul < 0 || a.Noul > 1 || res.Usage.InputTokens <= 0 || res.Usage.OutputTokens != 0 {
			return fmt.Errorf("invalid decision result: %+v", res)
		}
		return nil
	}},
})
