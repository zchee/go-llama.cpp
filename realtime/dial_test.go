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
	"net"
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

	tests := map[string]*DialOptions{
		"error: nil options":    nil,
		"error: with intent":    {Intent: IntentTranscription, Model: "m", ReadLimit: 1 << 10},
		"error: request option": {RequestOptions: []llama.RequestOption{llama.WithAutoload(true)}},
	}
	for name, opts := range tests {
		t.Run(name, func(t *testing.T) {
			s, err := Dial(t.Context(), c, opts)
			if err == nil {
				_ = s.CloseNow()
				t.Fatalf("Dial to a closed port succeeded")
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
