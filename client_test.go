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

package llama

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	gocmp "github.com/google/go-cmp/cmp"
)

const testAPIKey = "sk-test-secret-0123456789"

func TestNewValidatesBaseURL(t *testing.T) {
	tests := map[string]struct {
		baseURL  string
		wantErr  string
		wantBase string
	}{
		"success: host and port":             {baseURL: "http://127.0.0.1:8080", wantBase: "http://127.0.0.1:8080"},
		"success: api prefix trailing slash": {baseURL: "https://example.com/llama/", wantBase: "https://example.com/llama"},
		"error: empty":                       {baseURL: "", wantErr: "empty base URL"},
		"error: scheme":                      {baseURL: "ws://127.0.0.1:8080", wantErr: "scheme must be http or https"},
		"error: no host":                     {baseURL: "http:///x", wantErr: "no host"},
		"error: credentials":                 {baseURL: "http://u:p@127.0.0.1", wantErr: "must not carry credentials"},
		"error: query":                       {baseURL: "http://127.0.0.1/?a=b", wantErr: "query or fragment"},
		"error: unparsable":                  {baseURL: "http://[::1", wantErr: "invalid base URL"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, err := New(tt.baseURL)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("New(%q) error = %v, want one containing %q", tt.baseURL, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("New(%q): %v", tt.baseURL, err)
			}
			if got := c.base.String(); got != tt.wantBase {
				t.Errorf("base = %q, want %q", got, tt.wantBase)
			}
		})
	}
}

func TestNewDefaults(t *testing.T) {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
	if c.HTTPClient().Timeout != 0 {
		t.Errorf("default HTTP client Timeout = %v, want 0 so that streams are not cut", c.HTTPClient().Timeout)
	}
	if _, ok := c.HTTPClient().Transport.(*http.Transport); !ok {
		t.Errorf("default transport is %T, want *http.Transport", c.HTTPClient().Transport)
	}
	if c.HTTPClient().Transport == http.DefaultTransport {
		t.Errorf("the default transport must be a clone, not http.DefaultTransport itself")
	}
	if diff := gocmp.Diff(DefaultRetryPolicy(), c.retry); diff != "" {
		t.Errorf("retry policy (-want +got):\n%s", diff)
	}
	if c.maxEventSize != DefaultMaxEventSize || c.userAgent != DefaultUserAgent {
		t.Errorf("maxEventSize = %d, userAgent = %q", c.maxEventSize, c.userAgent)
	}
}

func TestNewOptions(t *testing.T) {
	hc := &http.Client{Timeout: time.Minute}
	policy := RetryPolicy{MaxAttempts: 7, InitialInterval: time.Second}
	c, err := New("http://127.0.0.1:8080",
		WithHTTPClient(hc), WithUserAgent("ua/1"), WithRetry(policy), WithMaxEventSize(1234), nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.HTTPClient() == hc {
		t.Errorf("the caller's client must be copied, not modified")
	}
	if c.HTTPClient().Timeout != hc.Timeout || hc.CheckRedirect != nil {
		t.Errorf("copy lost the caller's Timeout, or the caller's CheckRedirect was changed")
	}
	if c.userAgent != "ua/1" || c.maxEventSize != 1234 || c.retry != policy {
		t.Errorf("options not applied: userAgent=%q maxEventSize=%d retry=%+v", c.userAgent, c.maxEventSize, c.retry)
	}

	c, err = New("http://127.0.0.1:8080", WithMaxEventSize(-1))
	if err != nil {
		t.Fatal(err)
	}
	if c.maxEventSize != DefaultMaxEventSize {
		t.Errorf("WithMaxEventSize(-1) gave %d, want the default", c.maxEventSize)
	}

	if _, err := New("http://127.0.0.1:8080", WithAPIKeyHeader(APIKeyHeader(9))); err == nil {
		t.Errorf("New accepted an unknown APIKeyHeader")
	}
}

// TestNewWrappedDefaultTransport replaces a process global and must not run in parallel.
func TestNewWrappedDefaultTransport(t *testing.T) {
	original := http.DefaultTransport
	wrapper := &struct{ http.RoundTripper }{original}
	http.DefaultTransport = wrapper
	t.Cleanup(func() { http.DefaultTransport = original })
	c, err := New("http://localhost")
	if err != nil || c.HTTPClient().Transport != wrapper {
		t.Fatalf("wrapped default transport was not retained: %v", err)
	}
	for _, hc := range []*http.Client{nil, {}} {
		_, err := New("http://localhost", WithHTTPClient(hc), WithUnixSocket("server.sock"))
		if err == nil || !strings.Contains(err.Error(), "needs an *http.Transport") {
			t.Errorf("WithUnixSocket error = %v, want an unsupported transport error", err)
		}
	}
}

func TestWithUnixSocket(t *testing.T) {
	sock := filepath.Join(t.TempDir(), "absent.sock")
	tests := map[string]struct {
		hc      *http.Client
		wantErr string
	}{
		"success: default client":                   {},
		"success: custom client with nil transport": {hc: &http.Client{Timeout: time.Second}},
		"success: custom TLS dialers": {hc: &http.Client{Transport: &http.Transport{
			DialTLS:        new(net.Dialer).Dial, //nolint:staticcheck // Check caller-supplied legacy dialers too.
			DialTLSContext: new(net.Dialer).DialContext,
		}}},
		"success: custom *http.Transport":  {hc: &http.Client{Transport: &http.Transport{MaxIdleConns: 3}}},
		"error: transport of another type": {hc: &http.Client{Transport: roundTripperFunc(nil)}, wantErr: "needs an *http.Transport"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			opts := []ClientOption{WithUnixSocket(sock)}
			if tt.hc != nil {
				opts = append(opts, WithHTTPClient(tt.hc))
			}
			c, err := New("http://localhost", opts...)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("New error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			tr, ok := c.HTTPClient().Transport.(*http.Transport)
			if !ok {
				t.Fatalf("transport is %T", c.HTTPClient().Transport)
			}
			if tt.hc != nil && tt.hc.Transport != nil && tt.hc.Transport == tr {
				t.Errorf("the caller's transport was modified instead of cloned")
			}
			if tt.hc != nil && c.HTTPClient().Timeout != tt.hc.Timeout {
				t.Errorf("Timeout of the caller's client was not kept")
			}
			if tr.DialTLS != nil || tr.DialTLSContext != nil { //nolint:staticcheck // https must not reach a legacy dialer either.
				t.Error("TLS dialer can bypass the unix socket")
			}
			// Dialing any TCP address must go to the socket path instead.
			_, err = tr.DialContext(t.Context(), "tcp", "example.com:80")
			if err == nil || !strings.Contains(err.Error(), "unix") || !strings.Contains(err.Error(), sock) {
				t.Errorf("DialContext error = %v, want a unix dial of %s", err, sock)
			}
		})
	}
}

func TestClientNeverPrintsAPIKey(t *testing.T) {
	c, err := New("http://127.0.0.1:8080", WithAPIKey(testAPIKey))
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	logger.Info("client", "c", c)
	outputs := []string{
		fmt.Sprintf("%v", c), fmt.Sprintf("%+v", c), fmt.Sprintf("%#v", c), fmt.Sprintf("%s", c), c.String(), buf.String(),
	}
	for _, out := range outputs {
		if strings.Contains(out, testAPIKey) {
			t.Errorf("output contains the API key: %s", out)
		}
	}
	if !strings.Contains(c.String(), "apiKey: true") {
		t.Errorf("String() = %q, want it to say a key is set", c.String())
	}
}

// roundTripperFunc adapts a function to http.RoundTripper.
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	if f == nil {
		return nil, errors.New("not callable")
	}
	return f(r)
}
