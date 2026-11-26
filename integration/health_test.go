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
	"context"
	"errors"
	"net/http"
	"net/http/httptrace"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

// A wrong API prefix can point to a static file served by llama-server instead of its health endpoint.
func TestInvalidHealthResponse(t *testing.T) {
	dir := t.TempDir()
	srv := harness.StartT(t, harness.Options{Model: harness.Model(t, "stories260K"), Args: []string{"--ui", "--path", dir}})
	tests := map[string]struct{ body string }{
		"empty": {}, "not-json": {"<html>"}, "missing-status": {"{}"}, "unknown-status": {`{"status":"other"}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := filepath.Join(dir, name)
			if err := os.Mkdir(prefix, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(prefix, "health"), []byte(tt.body), 0o600); err != nil {
				t.Fatal(err)
			}
			base, obs := srv.Client(t)
			c, err := llama.New(srv.URL+"/"+name, llama.WithHTTPClient(base.HTTPClient()))
			if err != nil {
				t.Fatal(err)
			}
			if res, err := c.Health(t.Context()); err == nil || res != nil {
				t.Errorf("Health = %+v, %v; want an invalid response error", res, err)
			}
			obs.Reset()
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			if err := c.WaitReady(ctx); err == nil || errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("WaitReady = %v; want an invalid response error", err)
			}
			if ex := obs.Exchanges(); len(ex) != 1 || ex[0].StatusCode != http.StatusOK || !ex[0].BodyClosed() {
				t.Errorf("WaitReady exchanges = %+v; want one closed 200 response", ex)
			}
		})
	}
}

func TestWaitReadyRetriesResponseHeaderTimeout(t *testing.T) {
	srv := presetStories260K.server(t)
	tr := &http.Transport{ResponseHeaderTimeout: 10 * time.Millisecond}
	t.Cleanup(tr.CloseIdleConnections)
	c, err := llama.New(srv.URL, llama.WithHTTPClient(&http.Client{Transport: tr}))
	if err != nil {
		t.Fatal(err)
	}
	// Hold the first real response in net/http's read loop until the next poll starts.
	var polls atomic.Int32
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	defer unblock()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
		GetConn: func(string) {
			if polls.Add(1) > 1 {
				unblock()
			}
		},
		GotFirstResponseByte: func() { <-release },
	})
	if err := c.WaitReady(ctx); err != nil {
		t.Fatalf("WaitReady after %d polls: %v", polls.Load(), err)
	}
	if polls.Load() < 2 {
		t.Fatalf("WaitReady succeeded without polling again")
	}
}

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
