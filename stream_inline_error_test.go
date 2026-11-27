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
	"net/http"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
)

func TestInlineErrorMember(t *testing.T) {
	tests := map[string]struct {
		payload  string
		apiError bool
	}{
		"success: string is ordinary data":               {payload: `{"error":"boom"}`},
		"success: number is ordinary data":               {payload: `{"error":503}`},
		"success: array is ordinary data":                {payload: `{"error":[]}`},
		"success: invalid message type is ordinary data": {payload: `{"error":{"message":42}}`},
		"error: empty object":                            {payload: `{"error":{}}`, apiError: true},
		"error: unknown fields only":                     {payload: `{"error":{"detail":true}}`, apiError: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			header := http.Header{"X-Test": {"1"}}
			var want *APIError
			if tt.apiError {
				want = &APIError{StatusCode: http.StatusOK, Raw: []byte(tt.payload), Header: header}
			}
			if diff := gocmp.Diff(want, inlineError(tt.payload, http.StatusOK, header)); diff != "" {
				t.Errorf("inline error mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
