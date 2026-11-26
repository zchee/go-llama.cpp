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
	"bytes"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

// testAPIKey is the key the authentication tests start the server with.
const testAPIKey = "sk-integration-4f9d2c71b0e8"

// TestAuthentication starts a server with --api-key and checks both header forms, a missing and a wrong key, and
// that /health stays open. Every error and every log line is checked for the key.
func TestAuthentication(t *testing.T) {
	srv := harness.StartT(t, harness.Options{
		Model:  harness.Model(t, "stories260K"),
		Args:   []string{"--ctx-size", "512", "--parallel", "1"},
		APIKey: testAPIKey,
	})

	tests := map[string]struct {
		opts       []llama.ClientOption
		wantStatus int
		wantHeader string
	}{
		"error: no key":                 {wantStatus: http.StatusUnauthorized},
		"error: wrong key":              {opts: []llama.ClientOption{llama.WithAPIKey("sk-wrong")}, wantStatus: http.StatusUnauthorized, wantHeader: "Authorization"},
		"success: Authorization Bearer": {opts: []llama.ClientOption{llama.WithAPIKey(testAPIKey)}, wantStatus: http.StatusOK, wantHeader: "Authorization"},
		"success: X-Api-Key": {
			opts:       []llama.ClientOption{llama.WithAPIKey(testAPIKey), llama.WithAPIKeyHeader(llama.APIKeyHeaderXAPIKey)},
			wantStatus: http.StatusOK,
			wantHeader: "X-Api-Key",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			opts := append([]llama.ClientOption{llama.WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))}, tt.opts...)
			c, obs := srv.Client(t, opts...)

			_, err := llama.Call[map[string]jsontext.Value](t.Context(), c, http.MethodGet, "/props", nil)
			switch tt.wantStatus {
			case http.StatusOK:
				if err != nil {
					t.Fatalf("GET /props with a valid key: %v", err)
				}
			default:
				apiErr, ok := errors.AsType[*llama.APIError](err)
				if !ok {
					t.Fatalf("GET /props error = %v, want *llama.APIError", err)
				}
				if apiErr.StatusCode != tt.wantStatus || apiErr.Type != llama.ErrorTypeAuthentication {
					t.Errorf("APIError = %d %s, want %d %s", apiErr.StatusCode, apiErr.Type, tt.wantStatus, llama.ErrorTypeAuthentication)
				}
			}

			health, err := c.Health(t.Context())
			if err != nil || health.Status != "ok" {
				t.Errorf("/health must be open whatever the key: %+v, %v", health, err)
			}

			exchanges := obs.Exchanges()
			if len(exchanges) != 2 {
				t.Fatalf("observed %d requests, want 2 (no retry of a 401):\n  %s", len(exchanges), describe(exchanges))
			}
			if tt.wantHeader != "" && exchanges[0].Header.Get(tt.wantHeader) == "" {
				t.Errorf("request did not carry the %s header", tt.wantHeader)
			}
			for _, other := range []string{"Authorization", "X-Api-Key"} {
				if other != tt.wantHeader && exchanges[0].Header.Get(other) != "" {
					t.Errorf("request carried %s as well", other)
				}
			}
			for _, out := range []string{fmt.Sprint(err), fmt.Sprintf("%+v", err), fmt.Sprintf("%#v", c), logs.String()} {
				if strings.Contains(out, testAPIKey) {
					t.Errorf("the API key appears in output: %s", out)
				}
			}
		})
	}
}

// TestUnixSocket starts a server on a UNIX socket and reaches it through llama.WithUnixSocket.
func TestUnixSocket(t *testing.T) {
	srv := harness.StartT(t, harness.Options{
		Model:      harness.Model(t, "stories260K"),
		Args:       []string{"--ctx-size", "512", "--parallel", "1"},
		UnixSocket: true,
	})
	c, err := llama.New(srv.URL, llama.WithUnixSocket(srv.Socket))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.HTTPClient().CloseIdleConnections)
	if err := c.WaitReady(t.Context()); err != nil {
		t.Fatalf("WaitReady over %s: %v", srv.Socket, err)
	}
	res, err := c.Health(t.Context())
	if err != nil || res.Status != "ok" {
		t.Fatalf("Health over %s = %+v, %v", srv.Socket, res, err)
	}
}
