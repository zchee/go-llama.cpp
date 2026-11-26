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
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"

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

			health, healthErr := c.Health(t.Context())
			if healthErr != nil || health.Status != "ok" {
				t.Errorf("/health must be open whatever the key: %+v, %v", health, healthErr)
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
			// The request log records the HTTP status of the protected call, error responses included.
			var propsLine string
			for line := range strings.Lines(logs.String()) {
				if strings.Contains(line, "path=/props") {
					propsLine = line
				}
			}
			if !strings.Contains(propsLine, fmt.Sprintf("status=%d", tt.wantStatus)) {
				t.Errorf("request log line of /props lacks status=%d: %q", tt.wantStatus, propsLine)
			}
			sentKey := strings.TrimPrefix(exchanges[0].Header.Get(tt.wantHeader), "Bearer ")
			outputs := map[string]string{"log": logs.String(), "client": fmt.Sprintf("%#v", c)}
			for _, format := range []string{"%v", "%+v", "%#v", "%s", "%q"} {
				outputs[format] = fmt.Sprintf(format, err)
			}
			for source, out := range outputs {
				if strings.Contains(out, testAPIKey) || (sentKey != "" && strings.Contains(out, sentKey)) {
					t.Errorf("%s disclosed an API key", source)
				}
			}
		})
	}
}

type bodyFailureTransport struct {
	next    http.RoundTripper
	failure error
}

func (tr bodyFailureTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := tr.next.RoundTrip(req)
	if err == nil {
		resp.Body = struct {
			io.Reader
			io.Closer
		}{iotest.ErrReader(tr.failure), resp.Body}
	}
	return resp, err
}

func TestResponseBodyErrorRedaction(t *testing.T) {
	const key = "not-a-real-credential"
	srv := presetStories260K.server(t)
	tr := http.DefaultTransport.(*http.Transport).Clone()
	t.Cleanup(tr.CloseIdleConnections)
	failure := errors.New("Authorization: Bearer " + key)
	c, err := llama.New(srv.URL, llama.WithAPIKey(key), llama.WithHTTPClient(&http.Client{Transport: bodyFailureTransport{tr, failure}}))
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct{ call func(t *testing.T) error }{
		"error: typed response body": {func(t *testing.T) error { _, err := c.Health(t.Context()); return err }},
		"error: raw response body": {func(t *testing.T) error {
			resp, err := llama.CallRaw(t.Context(), c, http.MethodGet, "/health", nil)
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			_, err = io.ReadAll(resp.Body)
			return err
		}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			err := tt.call(t)
			if !errors.Is(err, failure) {
				t.Fatal("missing injected body failure")
			}
			if !strings.Contains(err.Error(), "[REDACTED]") || strings.Contains(err.Error(), key) {
				t.Error("response body error was not redacted")
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
