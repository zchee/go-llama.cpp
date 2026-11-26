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

package systemone

import (
	"testing"

	llama "github.com/zchee/go-llama.cpp"
)

func TestNilRequest(t *testing.T) {
	lc, err := llama.New("http://127.0.0.1:1")
	if err != nil {
		t.Fatal(err)
	}
	c := NewClient(lc)
	tests := map[string]struct {
		call func() error
		want string
	}{
		"error: Evaluate": {func() error { _, err := c.Evaluate(t.Context(), nil); return err }, "systemone: nil evaluate request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if err := tt.call(); err == nil || err.Error() != tt.want {
				t.Errorf("got %v, want %q before sending", err, tt.want)
			}
		})
	}
}

func TestNewClient(t *testing.T) {
	lc, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
	if got := NewClient(lc).Llama(); got != lc {
		t.Errorf("NewClient(c).Llama() = %v, want the client it was built from", got)
	}
}
