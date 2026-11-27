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

package realtime

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"strings"
	"testing"

	"github.com/coder/websocket"

	llama "github.com/zchee/go-llama.cpp"
)

// TestDialConnectionFailure dials a closed local port, which is what Dial meets when no server listens.
func TestDialConnectionFailure(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	c, err := llama.New("http://"+addr, llama.WithAPIKey("sk-realtime-secret"))
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()

	tests := map[string]struct{ opts *DialOptions }{
		"error: nil options":    {},
		"error: with intent":    {&DialOptions{Intent: IntentTranscription, Model: "m", ReadLimit: 1 << 10}},
		"error: request option": {&DialOptions{RequestOptions: []llama.RequestOption{llama.WithAutoload(true)}}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			s, err := Dial(t.Context(), c, tt.opts)
			if err == nil {
				_ = s.CloseNow()
				t.Fatalf("Dial to a closed port succeeded")
			}
			if op, ok := errors.AsType[*net.OpError](err); !ok || op.Op != "dial" {
				t.Errorf("error = %v, want a wrapped network dial error", err)
			}
			if !strings.HasPrefix(err.Error(), "realtime: dial:") {
				t.Errorf("error = %v, want a dial error", err)
			}
			if strings.Contains(err.Error(), "sk-realtime-secret") {
				t.Errorf("error leaks the API key: %v", err)
			}
		})
	}
}

func TestNormalizeCloseError(t *testing.T) {
	tests := map[string]struct {
		err     error
		wantNil bool
	}{
		"success: nil":            {err: nil, wantNil: true},
		"success: already closed": {err: net.ErrClosed, wantNil: true},
		"success: normal closure": {err: websocket.CloseError{Code: websocket.StatusNormalClosure}, wantNil: true},
		"error: abnormal closure": {err: websocket.CloseError{Code: websocket.StatusInternalError}},
		"error: other":            {err: errors.New("boom")},
		"error: context canceled": {err: context.Canceled},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := normalizeCloseError(tt.err); (got == nil) != tt.wantNil {
				t.Errorf("normalizeCloseError(%v) = %v, want nil %t", tt.err, got, tt.wantNil)
			}
		})
	}
}

// dumpError quotes the whole handshake request, headers included, as some transports and proxies do.
type dumpError struct {
	dump string
}

func (e *dumpError) Error() string { return "handshake failed: " + e.dump }

type dumpingTransport struct{}

func (dumpingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	dump, err := httputil.DumpRequest(r, false)
	if err != nil {
		return nil, err
	}
	return nil, &dumpError{dump: string(dump)}
}

// TestDialErrorsNeverCarryAPIKey sends the handshake through a transport whose error quotes the request: the
// handshake bypasses the llama client's request path, so Dial must redact its own errors.
func TestDialErrorsNeverCarryAPIKey(t *testing.T) {
	const key = "sk-realtime-handshake-secret"
	tests := map[string]struct{ header llama.APIKeyHeader }{
		"success: bearer":    {llama.APIKeyHeaderAuthorization},
		"success: X-Api-Key": {llama.APIKeyHeaderXAPIKey},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, err := llama.New("http://127.0.0.1:8080", llama.WithAPIKey(key), llama.WithAPIKeyHeader(tt.header),
				llama.WithHTTPClient(&http.Client{Transport: dumpingTransport{}}))
			if err != nil {
				t.Fatal(err)
			}
			_, err = Dial(t.Context(), c, &DialOptions{Intent: IntentTranscription})
			if err == nil {
				t.Fatalf("Dial through a failing transport succeeded")
			}
			inner, ok := errors.AsType[*dumpError](err)
			if !ok || !strings.Contains(inner.dump, key) {
				t.Fatalf("the transport error is not reachable or does not quote the key, so the test proves nothing: %v", err)
			}
			for _, out := range []string{err.Error(), fmt.Sprintf("%+v", err), fmt.Sprintf("%#v", err), fmt.Sprintf("%q", err)} {
				if strings.Contains(out, key) {
					t.Errorf("Dial error carries the API key: %s", out)
				}
			}
		})
	}
}
