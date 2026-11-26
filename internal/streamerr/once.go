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

// Package streamerr provides single-use error streams for invalid requests.
package streamerr

import (
	"iter"
	"sync/atomic"
)

// Once returns a stream that yields first on its first iteration and consumed on every later iteration.
func Once[T any](first, consumed error) iter.Seq2[*T, error] {
	var used atomic.Bool
	return func(yield func(*T, error) bool) {
		if used.Swap(true) {
			yield(nil, consumed)
			return
		}
		yield(nil, first)
	}
}
