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
	"encoding/json/jsontext"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// recorder captures the failures a helper reports, so that a test can assert that a helper fails.
type recorder struct {
	testing.TB
	mu     sync.Mutex
	errors []string
	fatal  bool
}

func (r *recorder) Helper() {}

func (r *recorder) Errorf(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}

func (r *recorder) Fatalf(format string, args ...any) {
	r.mu.Lock()
	r.errors = append(r.errors, fmt.Sprintf(format, args...))
	r.fatal = true
	r.mu.Unlock()
	runtime.Goexit()
}

// run calls fn with a recorder on its own goroutine so that Fatalf can stop it.
func run(t *testing.T, fn func(tb testing.TB)) *recorder {
	r := &recorder{TB: t}
	done := make(chan struct{})
	go func() {
		defer close(done)
		fn(r)
	}()
	<-done
	return r
}

type good struct {
	A     int                       `json:"a"`
	Inner *goodInner                `json:"inner,omitzero"`
	List  []goodInner               `json:"list,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

type goodInner struct {
	B     string                    `json:"b"`
	Extra map[string]jsontext.Value `json:",embed"`
}

type lossy struct {
	A     int                       `json:"a"`
	Inner *lossInner                `json:"inner,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

type lossInner struct {
	B string `json:"b"`
}

type embedded struct {
	Shared
	C     int                       `json:"c"`
	Extra map[string]jsontext.Value `json:",embed"`
}

type Shared struct {
	S int `json:"s"`
}

type shadowed struct {
	SharedWithExtra
	Extra map[string]jsontext.Value `json:",embed"`
}

type SharedWithExtra struct {
	S     int                       `json:"s"`
	Extra map[string]jsontext.Value `json:",embed"`
}

type onlyEmbeddedFallback struct {
	SharedWithExtra
	C int `json:"c"`
}

type wrongFallback struct {
	Extra jsontext.Value `json:",embed"`
}

const doc = `{"a":1,"inner":{"b":"x"},"list":[{"b":"y"},{"b":"z"}]}`

func TestRoundTrip(t *testing.T) {
	r := run(t, func(tb testing.TB) { RoundTrip[good](tb, []byte(doc)) })
	if len(r.errors) != 0 {
		t.Errorf("RoundTrip failed a type that keeps everything: %v", r.errors)
	}
	r = run(t, func(tb testing.TB) { RoundTrip[lossy](tb, []byte(`{"a":1,"inner":{"b":"x","c":2}}`)) })
	if len(r.errors) != 1 || !strings.Contains(r.errors[0], "does not round-trip") {
		t.Errorf("RoundTrip did not report the member dropped by the nested struct: %v", r.errors)
	}
	r = run(t, func(tb testing.TB) { RoundTrip[good](tb, []byte(`{"a":"x"}`)) })
	if !r.fatal {
		t.Errorf("RoundTrip did not stop on a decode error")
	}
}

func TestInjectUnknown(t *testing.T) {
	r := run(t, func(tb testing.TB) { InjectUnknown[good](tb, []byte(doc)) })
	if len(r.errors) != 0 {
		t.Errorf("InjectUnknown failed a type that keeps everything: %v", r.errors)
	}
	r = run(t, func(tb testing.TB) { InjectUnknown[lossy](tb, []byte(`{"a":1,"inner":{"b":"x"}}`)) })
	if len(r.errors) == 0 || !strings.Contains(r.errors[0], "/inner") {
		t.Errorf("InjectUnknown did not name the object that lost its member: %v", r.errors)
	}
	r = run(t, func(tb testing.TB) {
		InjectUnknown[lossy](tb, []byte(`{"a":1,"inner":{"b":"x"}}`), SkipObjects("/inner"))
	})
	if len(r.errors) != 0 {
		t.Errorf("SkipObjects did not exclude /inner: %v", r.errors)
	}
	r = run(t, func(tb testing.TB) { InjectUnknown[[]int](tb, []byte(`[1,2]`)) })
	if !r.fatal {
		t.Errorf("InjectUnknown did not stop on a document without objects")
	}
}

func TestNoUnknown(t *testing.T) {
	data := []byte(`{"a":1,"x":1,"inner":{"b":"x","y":2},"list":[{"b":"y","z":3}]}`)
	r := run(t, func(tb testing.TB) { NoUnknown[good](tb, data, nil) })
	if len(r.errors) != 1 {
		t.Fatalf("NoUnknown reported %d failures, want 1: %v", len(r.errors), r.errors)
	}
	for _, want := range []string{"/: x", "/inner: y", "/list/0: z"} {
		if !strings.Contains(r.errors[0], want) {
			t.Errorf("NoUnknown report lacks %q:\n%s", want, r.errors[0])
		}
	}
	r = run(t, func(tb testing.TB) {
		NoUnknown[good](tb, data, map[string]string{"": "test", "/inner": "test", "/list/*": "test"})
	})
	if len(r.errors) != 0 {
		t.Errorf("allow patterns did not apply: %v", r.errors)
	}
	r = run(t, func(tb testing.TB) { NoUnknown[good](tb, data, map[string]string{"/inner": ""}) })
	if !r.fatal {
		t.Errorf("NoUnknown accepted an allow entry without a reason")
	}
}

func TestCheckFallbacks(t *testing.T) {
	tests := map[string]struct {
		check func(tb testing.TB)
		want  []string
	}{
		"success: every struct has one":             {check: func(tb testing.TB) { CheckFallbacks[good](tb) }},
		"success: embedded struct without fallback": {check: func(tb testing.TB) { CheckFallbacks[embedded](tb) }},
		"success: the only fallback is in the embedded struct": {
			check: func(tb testing.TB) { CheckFallbacks[onlyEmbeddedFallback](tb) },
		},
		"success: exempt type": {
			check: func(tb testing.TB) { CheckFallbacks[lossy](tb, reflect.TypeFor[lossInner]()) },
		},
		"error: nested struct without fallback": {
			check: func(tb testing.TB) { CheckFallbacks[lossy](tb) },
			want:  []string{"jsontest.lossy.Inner (jsontest.lossInner) has no Extra"},
		},
		"error: embedded fallback shadowed by the parent": {
			check: func(tb testing.TB) { CheckFallbacks[shadowed](tb) },
			want:  []string{"embeds jsontest.SharedWithExtra, whose own fallback is shadowed"},
		},
		"error: fallback of the wrong type": {
			check: func(tb testing.TB) { CheckFallbacks[wrongFallback](tb) },
			want:  []string{"fallback must be map[string]jsontext.Value"},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			r := run(t, tt.check)
			if len(tt.want) == 0 {
				if len(r.errors) != 0 {
					t.Errorf("unexpected failures: %v", r.errors)
				}
				return
			}
			if len(r.errors) != 1 {
				t.Fatalf("got %d failures, want 1: %v", len(r.errors), r.errors)
			}
			for _, w := range tt.want {
				if !strings.Contains(r.errors[0], w) {
					t.Errorf("failure lacks %q:\n%s", w, r.errors[0])
				}
			}
		})
	}
}

func TestEventData(t *testing.T) {
	stream := []byte(":\n\nevent: a\ndata: {\"x\":1}\n\ndata: {\"y\":2}\n\ndata: [DONE]\n\n")
	got := EventData(t, stream)
	want := []Event{{Type: "a", Data: []byte(`{"x":1}`)}, {Data: []byte(`{"y":2}`)}}
	if diff := gocmp.Diff(want, got); diff != "" {
		t.Errorf("EventData (-want +got):\n%s", diff)
	}
}

func TestPointerHelpers(t *testing.T) {
	tests := map[string]struct {
		path, pattern string
		want          bool
	}{
		"match: exact":        {path: "/a/b", pattern: "/a/b", want: true},
		"match: wildcard":     {path: "/a/3/b", pattern: "/a/*/b", want: true},
		"no: length differs":  {path: "/a/b/c", pattern: "/a/*", want: false},
		"no: segment differs": {path: "/a/b", pattern: "/a/c", want: false},
		"match: root":         {path: "", pattern: "", want: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := skipped(tt.path, []string{tt.pattern}); got != tt.want {
				t.Errorf("skipped(%q, %q) = %t, want %t", tt.path, tt.pattern, got, tt.want)
			}
		})
	}
	v := map[string]any{"a/b": []any{map[string]any{"~": 1.0}}}
	if got := lookup(v, "/"+escapePointer("a/b")+"/0/"+escapePointer("~")); got != 1.0 {
		t.Errorf("lookup through escaped names = %v, want 1", got)
	}
	if got := lookup(v, "/missing/0"); got != nil {
		t.Errorf("lookup of a missing path = %v, want nil", got)
	}
}
