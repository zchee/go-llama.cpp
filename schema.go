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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"reflect"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// SchemaFor returns the JSON schema of T, for SamplingParams.JSONSchema, an OpenAI response_format or the
// parameters of a tool. The schema follows the rules of github.com/google/jsonschema-go: exported fields named by
// their json tags, fields with omitzero or omitempty optional, and no additional properties.
//
// The schema generator does not understand the "embed" option of encoding/json/v2, so SchemaFor rejects any T
// whose type graph contains a struct field tagged with it, whether the field is a fallback (a map or
// jsontext.Value collecting unknown members) or a struct whose members are flattened into the parent. Such types
// would get a schema that does not match their JSON form. The request and response types of this module carry
// such fallbacks and must not be passed to SchemaFor.
func SchemaFor[T any]() (jsontext.Value, error) {
	t := reflect.TypeFor[T]()
	if path, ok := findEmbed(t, map[reflect.Type]bool{}); ok {
		return nil, fmt.Errorf("llama: SchemaFor[%v]: field %s uses the json \"embed\" option, which the schema generator does not support", t, path)
	}
	s, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, fmt.Errorf("llama: SchemaFor[%v]: %w", t, err)
	}
	data, err := json.Marshal(s, jsonOptions)
	if err != nil {
		return nil, fmt.Errorf("llama: SchemaFor[%v]: encoding schema: %w", t, err)
	}
	return data, nil
}

// findEmbed searches the type graph of t for a struct field with the json "embed" option and returns the path of
// the first one found.
func findEmbed(t reflect.Type, seen map[reflect.Type]bool) (string, bool) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if seen[t] {
		return "", false
	}
	seen[t] = true

	switch t.Kind() {
	case reflect.Slice, reflect.Array:
		return findEmbed(t.Elem(), seen)
	case reflect.Map:
		return findEmbed(t.Elem(), seen)
	case reflect.Struct:
		for f := range t.Fields() {
			if !f.IsExported() && !f.Anonymous {
				continue
			}
			if hasJSONOption(f.Tag.Get("json"), "embed") {
				return t.String() + "." + f.Name, true
			}
			if path, ok := findEmbed(f.Type, seen); ok {
				return path, true
			}
		}
	}
	return "", false
}

// hasJSONOption reports whether a json struct tag carries option opt after its name.
func hasJSONOption(tag, opt string) bool {
	_, opts, ok := strings.Cut(tag, ",")
	if !ok {
		return false
	}
	for o := range strings.SplitSeq(opts, ",") {
		if o == opt {
			return true
		}
	}
	return false
}
