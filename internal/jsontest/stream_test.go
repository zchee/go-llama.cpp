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
	"fmt"
	"iter"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
)

func TestSingleErrorStream(t *testing.T) {
	invalid := errors.New("invalid request")
	consumed := errors.New("consumed")
	type pair struct {
		value *int
		err   error
	}
	first := []pair{{nil, invalid}}
	second := []pair{{nil, fmt.Errorf("stream: %w", consumed)}}
	tests := map[string]struct {
		stoppedFirst, drainedFirst []pair
		afterBreak, afterDrain     []pair
		want                       []string
	}{
		"success: wrapped consumed error": {first, []pair{{nil, errors.New(invalid.Error())}}, second, second, nil},
		"error: two first pairs":          {first, []pair{{nil, invalid}, {nil, consumed}}, second, second, []string{"drain: first range yielded 2 pairs, want 1"}},
		"error: first value and no error": {first, []pair{{new(7), nil}}, second, second, []string{"drain: first range yielded a non-nil value: 7", "drain: first range yielded a nil error"}},
		"error: wrong second pair":        {first, first, second, []pair{{new(8), invalid}}, []string{"drain: second range yielded a non-nil value: 8", "drain: second range error = invalid request, want consumed"}},
		"error: empty ranges":             {nil, nil, nil, nil, []string{"early stop: first range yielded 0 pairs, want 1", "early stop: second range yielded 0 pairs, want 1", "drain: first range yielded 0 pairs, want 1", "drain: second range yielded 0 pairs, want 1"}},
		"error: not consumed after break": {first, first, first, second, []string{"early stop: second range error = invalid request, want consumed"}},
		"error: different first errors":   {[]pair{{nil, errors.New("other request")}}, first, second, second, []string{`early stop error = "other request", drained error = "invalid request"`}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			next := tt.stoppedFirst
			newSeq := func() iter.Seq2[*int, error] {
				pairs := next
				next = tt.drainedFirst
				return func(yield func(*int, error) bool) {
					current := pairs
					pairs = tt.afterDrain
					for _, p := range current {
						if !yield(p.value, p.err) {
							pairs = tt.afterBreak
							return
						}
					}
				}
			}
			messages, fatal := Failures(t, func(tb testing.TB) {
				var want error
				if len(tt.drainedFirst) != 0 {
					want = tt.drainedFirst[0].err
				}
				if got := SingleErrorStream(tb, newSeq, consumed); !errors.Is(got, want) {
					t.Errorf("returned error = %v, want %v", got, want)
				}
			})
			if fatal {
				t.Error("SingleErrorStream reported a fatal failure")
			}
			if diff := gocmp.Diff(tt.want, messages); diff != "" {
				t.Errorf("failures (-want +got):\n%s", diff)
			}
		})
	}
}
