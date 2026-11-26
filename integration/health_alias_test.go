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
	"net/http"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
)

func TestHealthAlias(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	res, err := llama.Call[llama.HealthResponse](t.Context(), c, http.MethodGet, "/v1/health", nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != "ok" {
		t.Errorf("GET /v1/health status = %q, want ok", res.Status)
	}
}
