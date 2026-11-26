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
	"context"
	"errors"
	"log/slog"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

// The fixture was recorded from llama-server v1.1.0 with integration/cmd/record.
func TestHealthResponseWire(t *testing.T) {
	data := jsontest.ReadFile(t, "testdata/health/ok.resp.json")
	got := jsontest.RoundTrip[HealthResponse](t, data)
	jsontest.InjectUnknown[HealthResponse](t, data)
	jsontest.NoUnknown[HealthResponse](t, data, nil)
	jsontest.CheckFallbacks[HealthResponse](t)
	if got.Status != "ok" {
		t.Errorf("Status = %q, want %q", got.Status, "ok")
	}
}

// closedAddr returns the address of a local TCP port that nothing listens on.
func closedAddr(t *testing.T) string {
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

func TestHealthNeverRetries(t *testing.T) {
	var logs bytes.Buffer
	c, err := New("http://"+closedAddr(t),
		WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))),
		WithRetry(RetryPolicy{MaxAttempts: 5, InitialInterval: time.Millisecond, MaxInterval: time.Millisecond, Multiplier: 1}))
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()
	if _, err := c.Health(t.Context()); err == nil {
		t.Fatalf("Health of a closed port succeeded")
	}
	if !strings.Contains(logs.String(), "attempts=1") {
		t.Errorf("Health retried; it must report the current state at once:\n%s", logs.String())
	}
}

func TestWaitReadyKeepsPollingUntilTheContextEnds(t *testing.T) {
	var logs bytes.Buffer
	c, err := New("http://"+closedAddr(t), WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))))
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()
	ctx, cancel := context.WithTimeout(t.Context(), 400*time.Millisecond)
	defer cancel()
	err = c.WaitReady(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("WaitReady error = %v, want one matching context.DeadlineExceeded", err)
	}
	if !strings.Contains(err.Error(), "last error") {
		t.Errorf("WaitReady error does not carry the last connection error: %v", err)
	}
	if polls := strings.Count(logs.String(), "llama request"); polls < 2 {
		t.Errorf("WaitReady polled %d times in 400ms, want it to keep polling", polls)
	}
}
