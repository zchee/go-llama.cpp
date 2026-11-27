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
		first, second []pair
		want          []string
	}{
		"success: wrapped consumed error": {first, second, nil},
		"error: two first pairs":          {[]pair{{nil, invalid}, {nil, consumed}}, second, []string{"first range yielded 2 pairs, want 1"}},
		"error: first value and no error": {[]pair{{new(7), nil}}, second, []string{"first range yielded a non-nil value: 7", "first range yielded a nil error"}},
		"error: wrong second pair":        {first, []pair{{new(8), invalid}}, []string{"second range yielded a non-nil value: 8", "second range error = invalid request, want consumed"}},
		"error: empty ranges":             {nil, nil, []string{"first range yielded 0 pairs, want 1", "second range yielded 0 pairs, want 1"}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			pairs := tt.first
			seq := func(yield func(*int, error) bool) {
				current := pairs
				pairs = tt.second
				for _, p := range current {
					if !yield(p.value, p.err) {
						return
					}
				}
			}
			messages, fatal := Failures(t, func(tb testing.TB) {
				var want error
				if len(tt.first) != 0 {
					want = tt.first[0].err
				}
				if got := SingleErrorStream(tb, seq, consumed); !errors.Is(got, want) {
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
