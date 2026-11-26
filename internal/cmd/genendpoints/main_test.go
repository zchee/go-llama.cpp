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

package main

import (
	"fmt"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// readme builds a README with n plain endpoint headings plus one bold heading and some non-endpoint headings.
func readme(n int) string {
	var b strings.Builder
	b.WriteString("# llama-server\n\n### Common params\n\n## API Endpoints\n\n")
	b.WriteString("### **GET** `/props`: Get server global properties.\n\n")
	for i := range n {
		fmt.Fprintf(&b, "### POST `/e%d`: endpoint\n\ntext ### GET `/not-a-heading`\n\n", i)
	}
	b.WriteString("### WebSocket `/v1/realtime`: Realtime\n#### POST `/too-deep`\n### Model presets\n")
	return b.String()
}

func TestExtract(t *testing.T) {
	tests := map[string]struct {
		src     string
		wantErr string
		want    []string
	}{
		"success: bold method and WebSocket are kept": {
			src: readme(wantEndpoints - 2),
		},
		"error: one heading short": {
			src:     readme(wantEndpoints - 3),
			wantErr: fmt.Sprintf("found %d endpoint headings, want %d", wantEndpoints-1, wantEndpoints),
		},
		"error: one heading too many": {
			src:     readme(wantEndpoints - 1),
			wantErr: fmt.Sprintf("found %d endpoint headings, want %d", wantEndpoints+1, wantEndpoints),
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := extract([]byte(tt.src))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("extract error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != wantEndpoints {
				t.Fatalf("got %d lines, want %d", len(got), wantEndpoints)
			}
			if diff := gocmp.Diff([]string{"GET /props", "POST /e0"}, got[:2]); diff != "" {
				t.Errorf("first lines (-want +got):\n%s", diff)
			}
			if last := got[len(got)-1]; last != "WebSocket /v1/realtime" {
				t.Errorf("last line = %q, want the WebSocket heading", last)
			}
		})
	}
}
