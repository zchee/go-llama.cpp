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

package streamerr

import (
	"errors"
	"sync"
	"testing"
)

func TestOnceConcurrent(t *testing.T) {
	first, consumed := errors.New("invalid request"), errors.New("already consumed")
	seq := Once[int](first, consumed)
	var results [2]error
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := range results {
		wg.Go(func() {
			<-start
			for _, err := range seq {
				results[i] = err
				break
			}
		})
	}
	close(start)
	wg.Wait()
	if errors.Is(results[1], first) {
		results[0], results[1] = results[1], results[0]
	}
	if !errors.Is(results[0], first) || !errors.Is(results[1], consumed) {
		t.Errorf("concurrent iterations yielded %v; want one %v and one %v", results, first, consumed)
	}
}

func TestOnce(t *testing.T) {
	first := errors.New("invalid request")
	consumed := errors.New("already consumed")
	tests := map[string]struct{ stopEarly bool }{
		"success: exhaust first iteration": {},
		"success: break on first error":    {stopEarly: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			seq := Once[int](first, consumed)
			for i, want := range []error{first, consumed, consumed} {
				count := 0
				for v, err := range seq {
					count++
					if v != nil || !errors.Is(err, want) {
						t.Errorf("range %d yielded (%v, %v), want (nil, %v)", i, v, err, want)
					}
					if i == 0 && tt.stopEarly {
						break
					}
				}
				if count != 1 {
					t.Errorf("range %d yielded %d times, want 1", i, count)
				}
			}
		})
	}
}
