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

// Package extraguard checks for conflicts between typed fields and extra request members.
package extraguard

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
)

type requestField struct {
	path      []int
	name      string
	tagged    bool
	ambiguous bool
}

type requestFields struct {
	extra    requestField
	members  map[string]requestField
	children [][]int
}

var requestTypes sync.Map // reflect.Type -> *requestFields

// The encoder reports cycles in the body. Stop following the same reference here rather than recursing
// indefinitely before the encoder gets to see it. Include slice length: prefixes can share an address.
type requestVisit struct {
	parent *requestVisit
	typ    reflect.Type
	ptr    uintptr
	length int
}

// Check rejects extra members that name a typed field in the same request object.
func Check(body any) error {
	return walkRequest(reflect.ValueOf(body), nil, true)
}

func walkRequest(v reflect.Value, parent *requestVisit, root bool) error {
	if root && v.Kind() != reflect.Pointer && v.Kind() != reflect.Interface && v.Kind() != reflect.Struct {
		return nil
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return nil
		}
		visit := requestVisit{parent: parent, typ: v.Type(), ptr: v.Pointer()}
		if v.Kind() == reflect.Slice {
			visit.length = v.Len()
		}
		for p := parent; p != nil; p = p.parent {
			if visit.typ == p.typ && visit.ptr == p.ptr && visit.length == p.length {
				return nil
			}
		}
		parent = &visit
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		return walkRequest(v.Elem(), parent, root)
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			if err := walkRequest(v.Index(i), parent, false); err != nil {
				return err
			}
		}
	case reflect.Map:
		for it := v.MapRange(); it.Next(); {
			if err := walkRequest(it.Value(), parent, false); err != nil {
				return err
			}
		}
	case reflect.Struct:
		fields := requestFieldsFor(v.Type())
		if fields.extra.path != nil && !fields.extra.ambiguous {
			extra := fieldAt(v, fields.extra.path)
			if extra.Kind() == reflect.Pointer {
				extra = extra.Elem()
			}
			if extra.Kind() == reflect.Map && extra.Len() != 0 {
				for it := extra.MapRange(); it.Next(); {
					key := it.Key().String()
					if f, ok := fields.members[key]; ok && !f.ambiguous {
						return fmt.Errorf("llama: %s.%s contains %q; use field %s instead", v.Type(), fields.extra.name, key, f.name)
					}
				}
			}
		}
		for _, path := range fields.children {
			if err := walkRequest(fieldAt(v, path), parent, false); err != nil {
				return err
			}
		}
	}
	return nil
}

// Unlike FieldByIndex, fieldAt permits an absent embedded pointer.
func fieldAt(v reflect.Value, path []int) reflect.Value {
	for _, i := range path {
		if v.Kind() == reflect.Pointer {
			v = v.Elem()
			if !v.IsValid() {
				return v
			}
		}
		v = v.Field(i)
	}
	return v
}

func requestFieldsFor(t reflect.Type) *requestFields {
	if cached, ok := requestTypes.Load(t); ok {
		return cached.(*requestFields)
	}
	fields := &requestFields{members: make(map[string]requestField)}
	fields.collect(t, nil, nil)
	if f := fields.extra; f.path != nil && !f.ambiguous && mayContainRequest(t.FieldByIndex(f.path).Type) {
		fields.children = append(fields.children, f.path)
	}
	for _, member := range fields.members {
		if !member.ambiguous && mayContainRequest(t.FieldByIndex(member.path).Type) {
			fields.children = append(fields.children, member.path)
		}
	}
	cached, _ := requestTypes.LoadOrStore(t, fields)
	return cached.(*requestFields)
}

func mayContainRequest(t reflect.Type) bool {
	// A recursive named collection cannot be reduced to its innermost element type.
	var seen []reflect.Type
	for t.Kind() == reflect.Pointer || t.Kind() == reflect.Map || t.Kind() == reflect.Array || t.Kind() == reflect.Slice {
		if slices.Contains(seen, t) {
			return false
		}
		seen = append(seen, t)
		t = t.Elem()
	}
	return t.Kind() == reflect.Struct || t.Kind() == reflect.Interface
}

func (fields *requestFields) collect(t reflect.Type, path []int, parents []reflect.Type) {
	if slices.Contains(parents, t) {
		return
	}
	parents = append(parents, t)
	for sf := range t.Fields() {
		tag := sf.Tag.Get("json")
		if tag == "-" {
			continue
		}
		index := append(slices.Clone(path), sf.Index[0])
		if !sf.IsExported() && !sf.Anonymous {
			// The library's sum types marshal children stored in private fields (e.g. Prompt.object).
			if t.Implements(reflect.TypeFor[json.MarshalerTo]()) && mayContainRequest(sf.Type) {
				fields.children = append(fields.children, index)
			}
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		tagged := name != ""
		if strings.HasPrefix(tag, "'") {
			name = quotedRequestName(tag)
		}
		field := requestField{path: index, name: sf.Name, tagged: tagged}
		if sf.Anonymous && !tagged || tag == ",embed" {
			ft := sf.Type
			if ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				fields.collect(ft, index, parents)
			} else if ft == reflect.TypeFor[jsontext.Value]() || ft.Kind() == reflect.Map && ft.Key().Kind() == reflect.String {
				switch {
				case fields.extra.path == nil, len(index) < len(fields.extra.path):
					fields.extra = field
				case len(index) == len(fields.extra.path):
					fields.extra.ambiguous = true
				}
			}
			continue
		}
		if !tagged {
			name = sf.Name
		}
		// Match json/v2 member selection: prefer the shallowest field, then an explicitly named one.
		prev, ok := fields.members[name]
		switch {
		case !ok, len(index) < len(prev.path), len(index) == len(prev.path) && tagged && !prev.tagged:
			fields.members[name] = field
		case len(index) == len(prev.path) && tagged == prev.tagged:
			prev.ambiguous = true
			fields.members[name] = prev
		}
	}
}

// A quoted json/v2 member name can contain a comma or an escaped quote.
func quotedRequestName(tag string) string {
	var name strings.Builder
	for s := tag[1:]; len(s) != 0 && s[0] != '\''; {
		c, multi, rest, err := strconv.UnquoteChar(s, '\'')
		if err != nil {
			// Let the JSON encoder report the malformed tag.
			return ""
		}
		if multi {
			name.WriteRune(c)
		} else {
			name.WriteByte(byte(c))
		}
		s = rest
	}
	return name.String()
}
