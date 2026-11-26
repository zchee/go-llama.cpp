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

package extraguard

import (
	"reflect"
	"strconv"
	"testing"
)

func TestCheckDoesNotCacheAnonymousStructs(t *testing.T) {
	var cached int
	for i := range 128 {
		typ := reflect.StructOf([]reflect.StructField{{Name: "Field" + strconv.Itoa(i), Type: reflect.TypeFor[int]()}})
		if err := Check(reflect.New(typ).Interface()); err != nil {
			t.Fatal(err)
		}
		if _, ok := requestTypes.Load(typ); ok {
			cached++
		}
	}
	if cached != 0 {
		t.Errorf("guard retained metadata for %d runtime-generated types, want none", cached)
	}
}
