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

package jsontest

import "testing"

// StreamFrameCuts returns every recorded SSE prefix ending in two consecutive newlines.
// It fails the test if the recording itself does not end at such a boundary.
func StreamFrameCuts(t testing.TB, data []byte) [][]byte {
	t.Helper()
	var cuts [][]byte
	for i := 0; i+1 < len(data); i++ {
		if data[i] == '\n' && data[i+1] == '\n' {
			cuts = append(cuts, data[:i+2])
		}
	}
	if len(cuts) == 0 || len(cuts[len(cuts)-1]) != len(data) {
		t.Fatalf("recorded stream has %d frame boundaries and %d total bytes; want a final frame boundary", len(cuts), len(data))
	}
	return cuts
}
