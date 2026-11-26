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
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestTransportRetryableError(t *testing.T) {
	closedAddr := func(t *testing.T) string {
		t.Helper()
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		addr := ln.Addr().String()
		if err := ln.Close(); err != nil {
			t.Fatal(err)
		}
		return addr
	}

	assertNoKey := func(t *testing.T, err error) {
		t.Helper()
		for _, out := range []string{err.Error(), fmt.Sprintf("%v", err), fmt.Sprintf("%+v", err), fmt.Sprintf("%#v", err)} {
			if strings.Contains(out, testAPIKey) {
				t.Errorf("error output contains API key: %s", out)
			}
		}
	}

	tests := map[string]struct {
		run func(t *testing.T)
	}{
		"error: unreachable port via Call strips retryable marker and omits API key": {
			run: func(t *testing.T) {
				addr := closedAddr(t)
				c, err := New("http://"+addr, WithAPIKey(testAPIKey),
					WithRetry(RetryPolicy{MaxAttempts: 2, InitialInterval: time.Millisecond, MaxInterval: time.Millisecond, Multiplier: 1}))
				if err != nil {
					t.Fatal(err)
				}
				defer c.HTTPClient().CloseIdleConnections()

				_, err = Call[HealthResponse](t.Context(), c, http.MethodGet, "/health", nil)
				if err == nil {
					t.Fatal("Call to closed port succeeded")
				}
				if _, ok := errors.AsType[*retryableError](err); ok {
					t.Errorf("public Call returned internal *retryableError: %v", err)
				}
				if !strings.Contains(err.Error(), addr) {
					t.Errorf("Call error %q does not mention target address %q", err.Error(), addr)
				}
				assertNoKey(t, err)
			},
		},
		"error: canceled context via Call returns context error without API key": {
			run: func(t *testing.T) {
				addr := closedAddr(t)
				c, err := New("http://"+addr, WithAPIKey(testAPIKey),
					WithRetry(RetryPolicy{MaxAttempts: 2, InitialInterval: time.Millisecond, MaxInterval: time.Millisecond, Multiplier: 1}))
				if err != nil {
					t.Fatal(err)
				}
				defer c.HTTPClient().CloseIdleConnections()

				ctx, cancel := context.WithCancel(t.Context())
				cancel()
				_, err = Call[HealthResponse](ctx, c, http.MethodGet, "/health", nil)
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("Call error = %v, want context.Canceled", err)
				}
				if _, ok := errors.AsType[*retryableError](err); ok {
					t.Errorf("public Call returned internal *retryableError: %v", err)
				}
				assertNoKey(t, err)
			},
		},
		"error: attempt produces retryableError whose Error matches unwrapped cause and redacts API key": {
			run: func(t *testing.T) {
				addr := closedAddr(t)
				c, err := New("http://"+addr, WithAPIKey(testAPIKey))
				if err != nil {
					t.Fatal(err)
				}
				defer c.HTTPClient().CloseIdleConnections()

				req, err := c.NewRequest(t.Context(), http.MethodGet, "/health", nil)
				if err != nil {
					t.Fatal(err)
				}
				_, err = c.attempt(t.Context(), req, 1)
				re, ok := errors.AsType[*retryableError](err)
				if !ok {
					t.Fatalf("attempt error = %T (%v), want *retryableError", err, err)
				}
				if re.Error() == "" || re.Error() != re.Unwrap().Error() {
					t.Errorf("retryableError.Error() = %q, Unwrap().Error() = %q", re.Error(), re.Unwrap().Error())
				}
				assertNoKey(t, re)

				redacted := c.RedactError(&retryableError{err: fmt.Errorf("request to %s failed with %s", addr, testAPIKey)})
				if redacted == nil || !strings.Contains(redacted.Error(), redactedText) {
					t.Errorf("RedactError on retryableError = %v, want %q", redacted, redactedText)
				}
				assertNoKey(t, redacted)
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			tt.run(t)
		})
	}
}
