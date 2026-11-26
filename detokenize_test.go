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
	"fmt"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestDetokenizeWire(t *testing.T) {
	tests := map[string]struct {
		tokens []Token
		want   string
	}{
		"success: basic":        {[]Token{198, 164}, "á"},
		"success: invalid_utf8": {[]Token{198}, "�"},
		"success: empty":        {[]Token{}, ""},
		"success: absent":       {},
	}
	jsontest.CheckFallbacks[DetokenizeRequest](t)
	jsontest.CheckFallbacks[DetokenizeResponse](t)
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			fixture := "testdata/detokenize/" + name[len("success: "):]
			tokenizeCheckRequest(t, "/detokenize", fixture, DetokenizeRequest{Tokens: tt.tokens})
			got := tokenizeFixture[DetokenizeResponse](t, fixture+".resp.json")
			if diff := gocmp.Diff(tt.want, got.Content); diff != "" {
				t.Errorf("content differs (-want +got):\n%s", diff)
			}
		})
	}
}

func ExampleClient_Detokenize() {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		panic(err)
	}
	// Token IDs belong to the server's active vocabulary.
	res, err := c.Detokenize(context.Background(), &DetokenizeRequest{Tokens: []Token{198, 164}})
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Content)
}
