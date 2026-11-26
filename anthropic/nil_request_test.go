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

package anthropic

import (
	"iter"
	"net/http/httptrace"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
)

func TestNilRequiredRequest(t *testing.T) {
	base, err := llama.New("http://127.0.0.1:1")
	if err != nil {
		t.Fatal(err)
	}
	c := NewClient(base)
	ctx := httptrace.WithClientTrace(t.Context(), &httptrace.ClientTrace{GetConn: func(hostPort string) {
		t.Errorf("unexpected HTTP connection attempt to %s for a nil request", hostPort)
	}})
	requestError := func(_ any, err error) error { return err }
	tests := map[string]struct {
		got  error
		want string
	}{
		"error: Messages":       {requestError(c.Messages(ctx, nil)), "anthropic: nil messages request"},
		"error: MessagesStream": {streamError(t, c.MessagesStream(ctx, nil)), "anthropic: nil messages stream request"},
		"error: CountTokens":    {requestError(c.CountTokens(ctx, nil)), "anthropic: nil count tokens request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if tt.got == nil || tt.got.Error() != tt.want {
				t.Errorf("error = %v, want %q", tt.got, tt.want)
			}
		})
	}
}

func streamError[T any](t *testing.T, seq iter.Seq2[*T, error]) error {
	t.Helper()
	for v, err := range seq {
		if v != nil {
			t.Errorf("unexpected stream value for nil request: %v", v)
		}
		return err
	}
	return nil
}
