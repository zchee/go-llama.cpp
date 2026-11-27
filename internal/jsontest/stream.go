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

import (
	"errors"
	"iter"
	"testing"
)

// SingleErrorStream checks that seq yields only (nil, error) on its first range
// and only (nil, consumed) on its second, allowing wrapped consumed errors.
// It returns the first error so the caller can check the request-specific failure.
func SingleErrorStream[T any](t testing.TB, seq iter.Seq2[*T, error], consumed error) error {
	t.Helper()
	var first error
	count := 0
	for v, err := range seq {
		if count == 0 {
			first = err
		}
		count++
		if v != nil {
			t.Errorf("first range yielded a non-nil value: %v", *v)
		}
		if err == nil {
			t.Errorf("first range yielded a nil error")
		}
	}
	if count != 1 {
		t.Errorf("first range yielded %d pairs, want 1", count)
	}
	count = 0
	for v, err := range seq {
		count++
		if v != nil {
			t.Errorf("second range yielded a non-nil value: %v", *v)
		}
		if !errors.Is(err, consumed) {
			t.Errorf("second range error = %v, want %v", err, consumed)
		}
	}
	if count != 1 {
		t.Errorf("second range yielded %d pairs, want 1", count)
	}
	return first
}

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
