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

// Package jsontest checks that wire types keep JSON members they do not name. It is imported only by tests.
//
// Every struct that maps a JSON object of a request or a response carries
//
//	Extra map[string]jsontext.Value `json:",embed"`
//
// so that unknown members survive a decode and encode round trip at every level. A test of a wire type T uses
// the helpers of this package on bytes recorded from a real server:
//
//	func TestCompletionResponseWire(t *testing.T) {
//		data := jsontest.ReadFile(t, "testdata/completion/basic.resp.json")
//		jsontest.RoundTrip[llama.CompletionResponse](t, data)        // decode, encode, compare semantically
//		jsontest.InjectUnknown[llama.CompletionResponse](t, data)    // an unknown member in every object survives
//		jsontest.NoUnknown[llama.CompletionResponse](t, data, nil)   // the type names every member the server sent
//		jsontest.CheckFallbacks[llama.CompletionResponse](t)         // every reachable struct has the fallback
//	}
//
// For a recorded event stream, EventData splits the stream into the data of its events, and each one is checked
// the same way with the event's type.
//
// Comparisons keep JSON numbers exact (9007199254740993 does not equal 9007199254740992), except that two
// non-integers that round to the same finite, nonzero float64 are equal: the server does not always print the
// shortest spelling of a double. Failures runs a helper against a recording testing.TB, for tests that show
// a helper rejects a type.
//
// A struct embedded without a JSON name (for example llama.SamplingParams) is part of its parent object and must
// not carry a fallback of its own: encoding/json/v2 lets the shallowest fallback win, so a nested one would never
// be filled. Types with their own JSON methods (the sum types) are skipped by CheckFallbacks and covered by
// InjectUnknown on fixtures that contain their object forms.
package jsontest

import (
	"bytes"
	"crypto/rand"
	"encoding"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"math/big"
	"os"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	sse "github.com/tmaxmax/go-sse"
)

// options are the JSON options of the llama packages.
var options = json.Deterministic(true)

// Failures runs check with a testing.TB that records failures instead of reporting them, and returns the recorded
// messages and whether check stopped with Fatal. Tests use it to show that a helper of this package rejects a
// type, for example that NoUnknown finds a member a sum type drops.
func Failures(t *testing.T, check func(tb testing.TB)) (messages []string, fatal bool) {
	t.Helper()
	r := &recorder{TB: t}
	done := make(chan struct{})
	go func() {
		defer close(done)
		check(r)
	}()
	<-done
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.messages), r.fatal
}

// recorder is a testing.TB that records Errorf and Fatalf; Fatalf ends the goroutine like testing.T does.
type recorder struct {
	testing.TB
	mu       sync.Mutex
	messages []string
	fatal    bool
}

func (r *recorder) Helper() {}

func (r *recorder) Errorf(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.messages = append(r.messages, fmt.Sprintf(format, args...))
}

func (r *recorder) Fatalf(format string, args ...any) {
	r.mu.Lock()
	r.messages = append(r.messages, fmt.Sprintf(format, args...))
	r.fatal = true
	r.mu.Unlock()
	runtime.Goexit()
}

// ReadFile returns the content of a fixture file and fails the test when it cannot be read.
func ReadFile(t testing.TB, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}
	return data
}

// Event is one event of a recorded server-sent event stream.
type Event struct {
	// Type is the event name, empty for unnamed events.
	Type string
	// Data is the event payload.
	Data []byte
}

// EventData parses a recorded server-sent event stream and returns its events, without the "[DONE]" terminator
// and without comments.
func EventData(t testing.TB, stream []byte) []Event {
	t.Helper()
	var out []Event
	for ev, err := range sse.Read(bytes.NewReader(stream), &sse.ReadConfig{MaxEventSize: 64 << 20}) {
		if err != nil {
			t.Fatalf("parsing event stream: %v", err)
		}
		if ev.Data == "" || ev.Data == "[DONE]" {
			continue
		}
		out = append(out, Event{Type: ev.Type, Data: []byte(ev.Data)})
	}
	return out
}

// RoundTrip decodes data into a T, encodes the result, and fails the test unless the output is semantically equal
// to data (same members and values; member order and number spelling may differ). It returns the decoded value.
func RoundTrip[T any](t testing.TB, data []byte) *T {
	t.Helper()
	v := new(T)
	if err := json.Unmarshal(data, v, options); err != nil {
		t.Fatalf("decoding into %T: %v\ninput: %s", v, err, data)
	}
	out, err := json.Marshal(v, options)
	if err != nil {
		t.Fatalf("encoding %T: %v", v, err)
	}
	if diff := gocmp.Diff(generic(t, data), generic(t, out)); diff != "" {
		t.Errorf("%T does not round-trip (-input +re-encoded):\n%s\ninput:      %s\nre-encoded: %s", v, diff, data, out)
	}
	return v
}

// A different name each run keeps a modeled member from standing in for the fallback.
var injectedName = "x_jsontest_unknown_" + rand.Text()

// InjectOption configures InjectUnknown.
type InjectOption func(*injectConfig)

type injectConfig struct {
	skip []string
}

// SkipObjects excludes objects from injection by JSON pointer pattern, where "*" matches one reference token, for
// example "/logit_bias" or "/choices/*/logprobs/content/*/top_logprobs/*". Use it for objects that are JSON maps
// rather than records (a map from token to bias has no place for an unknown member) and say why at the call site.
func SkipObjects(patterns ...string) InjectOption {
	return func(c *injectConfig) { c.skip = append(c.skip, patterns...) }
}

// InjectUnknown adds a synthetic member to every JSON object of data, at every depth and inside arrays, decodes the
// result into a T, encodes it again and fails the test for every object whose synthetic member did not survive.
func InjectUnknown[T any](t testing.TB, data []byte, opts ...InjectOption) {
	t.Helper()
	cfg := &injectConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	tree := generic(t, data)
	var paths []string
	inject(tree, "", cfg, &paths)
	if len(paths) == 0 {
		t.Fatalf("no JSON object to inject into: %s", data)
	}
	injected, err := json.Marshal(tree, options)
	if err != nil {
		t.Fatalf("encoding injected document: %v", err)
	}

	v := new(T)
	if err := json.Unmarshal(injected, v, options); err != nil {
		t.Fatalf("decoding injected document into %T: %v\ninput: %s", v, err, injected)
	}
	out, err := json.Marshal(v, options)
	if err != nil {
		t.Fatalf("encoding %T with probe %q: %v", v, injectedName, err)
	}
	got := generic(t, out)

	var lost []string
	for _, p := range paths {
		obj, ok := lookup(got, p).(map[string]any)
		if !ok || obj[injectedName] != p {
			lost = append(lost, p)
		}
	}
	if len(lost) > 0 {
		t.Errorf("%T dropped the unknown member %q injected into %d object(s):\n  %s\nthe struct decoding each of these objects needs an Extra map[string]jsontext.Value `json:\",embed\"` field", v, injectedName, len(lost), strings.Join(lost, "\n  "))
	}
	if diff := gocmp.Diff(tree, got); diff != "" {
		t.Errorf("%T changed the document with probe %q (-injected +re-encoded):\n%s", v, injectedName, diff)
	}
}

// inject adds the synthetic member to every object of v and records the JSON pointer of each.
func inject(v any, path string, cfg *injectConfig, paths *[]string) {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			inject(e, path+"/"+escapePointer(k), cfg, paths)
		}
		if !skipped(path, cfg.skip) {
			x[injectedName] = path
			*paths = append(*paths, path)
		}
	case []any:
		for i, e := range x {
			inject(e, path+"/"+strconv.Itoa(i), cfg, paths)
		}
	}
}

func escapePointer(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~", "~0"), "/", "~1")
}

func unescapePointer(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "~1", "/"), "~0", "~")
}

// skipped reports whether path matches one of the patterns.
func skipped(path string, patterns []string) bool {
	segs := strings.Split(path, "/")
	for _, p := range patterns {
		ps := strings.Split(p, "/")
		if len(ps) != len(segs) {
			continue
		}
		match := true
		for i := range ps {
			if ps[i] != "*" && ps[i] != segs[i] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// lookup returns the value at a JSON pointer, or nil.
func lookup(v any, path string) any {
	if path == "" {
		return v
	}
	for seg := range strings.SplitSeq(path[1:], "/") {
		switch x := v.(type) {
		case map[string]any:
			v = x[unescapePointer(seg)]
		case []any:
			i, err := strconv.Atoi(seg)
			if err != nil || i < 0 || i >= len(x) {
				return nil
			}
			v = x[i]
		default:
			return nil
		}
	}
	return v
}

// generic decodes JSON into maps, slices and scalars for a semantic comparison. It preserves number spellings so
// number.Equal can tell apart values a Go field cannot hold, such as an integer beyond float64 precision or a
// fraction narrowed by a float32.
func generic(t testing.TB, data []byte) any {
	t.Helper()
	dec := jsontext.NewDecoder(bytes.NewReader(data))
	v, err := decodeLossless(dec)
	if err == nil {
		if _, extra := dec.ReadToken(); !errors.Is(extra, io.EOF) {
			err = errors.New("data after the top-level value")
		}
	}
	if err != nil {
		t.Fatalf("decoding JSON: %v\ninput: %s", err, data)
	}
	return v
}

// decodeLossless reads one JSON value into map[string]any, []any, string, bool, nil or number.
func decodeLossless(dec *jsontext.Decoder) (any, error) {
	switch dec.PeekKind() {
	case '{':
		if _, err := dec.ReadToken(); err != nil {
			return nil, err
		}
		obj := map[string]any{}
		for dec.PeekKind() != '}' {
			k, err := dec.ReadToken()
			if err != nil {
				return nil, err
			}
			name := k.String()
			v, err := decodeLossless(dec)
			if err != nil {
				return nil, err
			}
			obj[name] = v
		}
		_, err := dec.ReadToken()
		return obj, err
	case '[':
		if _, err := dec.ReadToken(); err != nil {
			return nil, err
		}
		arr := []any{}
		for dec.PeekKind() != ']' {
			v, err := decodeLossless(dec)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		_, err := dec.ReadToken()
		return arr, err
	case '0':
		v, err := dec.ReadValue()
		return number(v), err
	}
	tok, err := dec.ReadToken()
	if err != nil {
		return nil, err
	}
	switch tok.Kind() {
	case 'n':
		return nil, nil
	case 't', 'f':
		return tok.Bool(), nil
	case '"':
		return tok.String(), nil
	default:
		return nil, fmt.Errorf("unexpected JSON token %v", tok.Kind())
	}
}

// number is a JSON number as written. Comparisons are exact: 1, 1.0 and 1e0 are equal while 9007199254740993
// and 9007199254740992 are not. Non-integers may also compare equal when they round to the same finite, nonzero
// float64 because the server does not always print the shortest spelling of a double. That treats every
// non-integer as a printed double, which holds for llama-server. Integer values get no such allowance, so two
// spellings of one double at or above 2^53 (9.999999999999999e+22 and 1e+23) stay unequal.
type number string

// Equal reports whether two numbers are the same rational or, when neither is an integer, the same finite
// nonzero float64; go-cmp uses it.
func (n number) Equal(m number) bool {
	a, okA := new(big.Rat).SetString(string(n))
	b, okB := new(big.Rat).SetString(string(m))
	if !okA || !okB {
		return n == m
	}
	if a.Cmp(b) == 0 {
		return true
	}
	if a.IsInt() || b.IsInt() {
		return false
	}
	x, errX := strconv.ParseFloat(string(n), 64)
	y, errY := strconv.ParseFloat(string(m), 64)
	// Overflow reports an error, but underflow can round a nonzero fraction to zero without one.
	return errX == nil && errY == nil && x != 0 && x == y
}

// MarshalJSONTo writes the number as written.
func (n number) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteValue(jsontext.Value(n))
}

// NoUnknown decodes data into a T and fails the test for every Extra fallback that received members, that is,
// for every member the server sent that the type does not name. Types with their own JSON methods (sum types such
// as llama.Prompt or openai.MessageContent) are searched through all their fields, unexported ones included. allow maps a JSON pointer pattern (as in
// SkipObjects) to the reason its members are intentionally left untyped.
func NoUnknown[T any](t testing.TB, data []byte, allow map[string]string) {
	t.Helper()
	v := new(T)
	if err := json.Unmarshal(data, v, options); err != nil {
		t.Fatalf("decoding into %T: %v\ninput: %s", v, err, data)
	}
	patterns := make([]string, 0, len(allow))
	for p, reason := range allow {
		if reason == "" {
			t.Fatalf("allow entry %q needs a reason", p)
		}
		patterns = append(patterns, p)
	}
	var found []string
	walkExtra(reflect.ValueOf(v).Elem(), "", func(path string, keys []string) {
		if !skipped(path, patterns) {
			found = append(found, fmt.Sprintf("%s: %s", pathOrRoot(path), strings.Join(keys, ", ")))
		}
	})
	if len(found) > 0 {
		t.Errorf("%T left server members untyped (JSON pointer: member names):\n  %s", v, strings.Join(found, "\n  "))
	}
}

func pathOrRoot(p string) string {
	if p == "" {
		return "/"
	}
	return p
}

// walkExtra calls fn for every non-empty embed fallback reachable from v, with the JSON pointer of its object.
func walkExtra(v reflect.Value, path string, fn func(path string, keys []string)) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			walkExtra(v.Elem(), path, fn)
		}
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			walkExtra(v.Index(i), path+"/"+strconv.Itoa(i), fn)
		}
	case reflect.Map:
		if v.Type().Key().Kind() != reflect.String {
			return
		}
		for _, k := range v.MapKeys() {
			walkExtra(v.MapIndex(k), path+"/"+escapePointer(k.String()), fn)
		}
	case reflect.Struct:
		walkStruct(v, path, fn)
	}
}

func walkStruct(v reflect.Value, path string, fn func(path string, keys []string)) {
	t := v.Type()
	opaque := customCodec(t)
	for i := range t.NumField() {
		f := t.Field(i)
		fv := v.Field(i)
		tag := f.Tag.Get("json")
		name, opts, _ := strings.Cut(tag, ",")
		switch {
		case !opaque && name == "-" && opts == "":
			continue
		case hasOption(opts, "embed") && isFallback(f.Type):
			if fv.Len() > 0 {
				keys := make([]string, 0, fv.Len())
				for _, k := range fv.MapKeys() {
					keys = append(keys, k.String())
				}
				sort.Strings(keys)
				fn(path, keys)
			}
		case opaque:
			// A sum codec can encode even a private branch at the object's own position.
			walkExtra(fv, path, fn)
		case !f.IsExported() && !f.Anonymous:
			continue
		case f.Anonymous && name == "", hasOption(opts, "embed"):
			walkExtra(fv, path, fn)
		default:
			if name == "" {
				name = f.Name
			}
			walkExtra(fv, path+"/"+escapePointer(name), fn)
		}
	}
}

func hasOption(opts, opt string) bool {
	return slices.Contains(strings.Split(opts, ","), opt)
}

var jsontextValue = reflect.TypeFor[jsontext.Value]()

// isFallback reports whether t can be an embed fallback that CheckFallbacks accepts.
func isFallback(t reflect.Type) bool {
	return t.Kind() == reflect.Map && t.Key().Kind() == reflect.String && t.Elem() == jsontextValue
}

var (
	marshalerTo     = reflect.TypeFor[json.MarshalerTo]()
	unmarshalerFrom = reflect.TypeFor[json.UnmarshalerFrom]()
	marshalerV1     = reflect.TypeFor[json.Marshaler]()
	unmarshalerV1   = reflect.TypeFor[json.Unmarshaler]()
	textMarshaler   = reflect.TypeFor[encoding.TextMarshaler]()
)

// customCodec reports whether t or *t has its own JSON or text methods.
func customCodec(t reflect.Type) bool {
	for _, x := range []reflect.Type{t, reflect.PointerTo(t)} {
		if x.Implements(marshalerTo) || x.Implements(unmarshalerFrom) || x.Implements(marshalerV1) ||
			x.Implements(unmarshalerV1) || x.Implements(textMarshaler) {
			return true
		}
	}
	return false
}

// CheckFallbacks fails the test for every struct reachable from T that maps a JSON object but has no
// Extra map[string]jsontext.Value `json:",embed"` fallback, and for every struct embedded without a JSON name that
// has a fallback of its own (it would never be filled). Types with their own JSON methods are not descended into.
// allow lists types that are intentionally exempt; give the reason in a comment at the call site.
func CheckFallbacks[T any](t testing.TB, allow ...reflect.Type) {
	t.Helper()
	exempt := map[reflect.Type]bool{}
	for _, a := range allow {
		exempt[a] = true
	}
	var problems []string
	checkType(reflect.TypeFor[T](), reflect.TypeFor[T]().String(), map[reflect.Type]bool{}, exempt, &problems)
	if len(problems) > 0 {
		t.Errorf("wire types without a usable unknown-member fallback:\n  %s", strings.Join(problems, "\n  "))
	}
}

func checkType(t reflect.Type, path string, seen, exempt map[reflect.Type]bool, problems *[]string) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if seen[t] || exempt[t] || t == jsontextValue {
		return
	}
	seen[t] = true
	if customCodec(t) {
		return
	}
	switch t.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		checkType(t.Elem(), path+"[]", seen, exempt, problems)
	case reflect.Struct:
		own, inlined := checkStruct(t, path, seen, exempt, problems, map[reflect.Type]bool{t: true})
		switch {
		case own == 0 && len(inlined) == 0:
			*problems = append(*problems, fmt.Sprintf("%s (%v) has no Extra map[string]jsontext.Value `json:\",embed\"` field", path, t))
		case own > 0 && len(inlined) > 0:
			*problems = append(*problems, fmt.Sprintf("%s (%v) embeds %s, whose own fallback is shadowed by the parent's and never filled; remove it", path, t, strings.Join(inlined, ", ")))
		case own+len(inlined) > 1:
			*problems = append(*problems, fmt.Sprintf("%s (%v) has more than one fallback", path, t))
		}
	}
}

// checkStruct checks the fields of t. It returns the number of fallbacks declared by t itself and the names of the
// embedded structs that declare one. inlining holds the structs being flattened into the current object, so that a
// struct embedding itself through a pointer (type Node struct{ *Node }) is visited once.
func checkStruct(t reflect.Type, path string, seen, exempt map[reflect.Type]bool, problems *[]string, inlining map[reflect.Type]bool) (own int, inlined []string) {
	for f := range t.Fields() {
		name, opts, _ := strings.Cut(f.Tag.Get("json"), ",")
		if name == "-" && opts == "" {
			continue
		}
		embedded := hasOption(opts, "embed") || (f.Anonymous && name == "")
		switch {
		case embedded && isFallback(f.Type):
			own++
		case embedded && hasOption(opts, "embed") && (f.Type == jsontextValue || f.Type.Kind() == reflect.Map):
			*problems = append(*problems, fmt.Sprintf("%s.%s: fallback must be map[string]jsontext.Value, got %v", path, f.Name, f.Type))
			own++
		case embedded:
			ft := f.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() != reflect.Struct || customCodec(ft) || inlining[ft] {
				continue
			}
			inlining[ft] = true
			n, nested := checkStruct(ft, path, seen, exempt, problems, inlining)
			delete(inlining, ft)
			if n > 0 {
				inlined = append(inlined, ft.String())
			}
			inlined = append(inlined, nested...)
		case !f.IsExported():
			continue
		default:
			checkType(f.Type, path+"."+f.Name, seen, exempt, problems)
		}
	}
	return own, inlined
}
