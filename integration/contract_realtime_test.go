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
	"net/url"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/realtime"
)

var _ = registerContracts(map[string]contractCase{
	"WebSocket /v1/realtime": {server: presetRealtimeOmni, query: url.Values{"intent": {realtime.IntentTranscription}}, call: func(t *testing.T, c *llama.Client) error {
		s, err := realtime.Dial(t.Context(), c, &realtime.DialOptions{Intent: realtime.IntentTranscription})
		if err != nil {
			return err
		}
		return s.Close(t.Context())
	}},
})
