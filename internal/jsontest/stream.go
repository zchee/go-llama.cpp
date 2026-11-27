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

// SingleErrorStream stops one sequence from newSeq after its first (nil, error)
// pair and drains another, checking that it yields exactly one such pair with
// the same error message. Both must then yield only (nil, consumed), allowing
// wrapped consumed errors. It returns the drained sequence's first error.
func SingleErrorStream[T any](t testing.TB, newSeq func() iter.Seq2[*T, error], consumed error) error {
	t.Helper()
	var first [2]error
	for i, mode := range [...]string{"early stop", "drain"} {
		seq := newSeq()
		count := 0
		for v, err := range seq {
			if count == 0 {
				first[i] = err
			}
			count++
			if v != nil {
				t.Errorf("%s: first range yielded a non-nil value: %v", mode, *v)
			}
			if err == nil {
				t.Errorf("%s: first range yielded a nil error", mode)
			}
			if i == 0 {
				break
			}
		}
		if count != 1 {
			t.Errorf("%s: first range yielded %d pairs, want 1", mode, count)
		}
		count = 0
		for v, err := range seq {
			count++
			if v != nil {
				t.Errorf("%s: second range yielded a non-nil value: %v", mode, *v)
			}
			if !errors.Is(err, consumed) {
				t.Errorf("%s: second range error = %v, want %v", mode, err, consumed)
			}
		}
		if count != 1 {
			t.Errorf("%s: second range yielded %d pairs, want 1", mode, count)
		}
	}
	stopped, drained := first[0], first[1]
	if stopped != nil && drained != nil && stopped.Error() != drained.Error() {
		t.Errorf("early stop error = %q, drained error = %q", stopped, drained)
	}
	return drained
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
