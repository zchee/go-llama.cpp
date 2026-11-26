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
	"errors"
	"net/http"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
)

// TestWaitReadyStopsOnPermanentErrors points a client at a path prefix the server does not serve: /health answers
// 404, which waiting cannot fix, so WaitReady must return at once.
func TestWaitReadyStopsOnPermanentErrors(t *testing.T) {
	srv := presetStories260K.server(t)
	c, err := llama.New(srv.URL + "/no-such-prefix")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.HTTPClient().CloseIdleConnections)
	start := time.Now()
	err = c.WaitReady(t.Context())
	apiErr, ok := errors.AsType[*llama.APIError](err)
	if !ok || apiErr.StatusCode != http.StatusNotFound || apiErr.Type != llama.ErrorTypeNotFound {
		t.Fatalf("WaitReady error = %v, want a 404 not_found_error", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("WaitReady took %v on a permanent error", elapsed)
	}
}
