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
	json "encoding/json/v2"
	"reflect"
	"strings"
	"testing"
	"time"

	gocmp "github.com/google/go-cmp/cmp"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// run calls check with a recorder and returns it, for the tests below.
func run(t *testing.T, check func(tb testing.TB)) *result {
	msgs, fatal := Failures(t, check)
	return &result{errors: msgs, fatal: fatal}
}

type result struct {
	errors []string
	fatal  bool
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

type fallback struct {
	Extra map[string]jsontext.Value `json:",embed"`
}
type (
	left    struct{ fallback }
	right   struct{ fallback }
	diamond struct {
		left
		right
	}
)

type opaqueRecord goodInner

func (r *opaqueRecord) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return json.UnmarshalDecode(dec, (*goodInner)(r))
}

type wrongFallback struct {
	Extra jsontext.Value `json:",embed"`
}

const doc = `{"a":1,"inner":{"b":"x"},"list":[{"b":"y"},{"b":"z"}]}`

func TestCheckWire(t *testing.T) {
	if got := CheckWire[good](t, []byte(doc)); got.A != 1 {
		t.Errorf("decoded a = %d, want 1", got.A)
	}
	messages, _ := Failures(t, func(tb testing.TB) {
		CheckWire[lossy](tb, []byte(`{"a":1,"untyped":2,"inner":{"b":"x","dropped":3}}`))
	})
	report := strings.Join(messages, "\n")
	for _, want := range []string{"does not round-trip", "dropped the unknown member", "/: untyped"} {
		if !strings.Contains(report, want) {
			t.Errorf("missing %q in failures:\n%s", want, report)
		}
	}
}

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

func TestRoundTripFloat64Spelling(t *testing.T) {
	type timing struct {
		PromptPerSecond float64 `json:"prompt_per_second"`
	}
	RoundTrip[timing](t, []byte(`{"prompt_per_second":5988.0239520958085}`))
}

func TestRoundTripFloat32PrecisionLoss(t *testing.T) {
	type value struct {
		N float32 `json:"n"`
	}
	r := run(t, func(tb testing.TB) { RoundTrip[value](tb, []byte(`{"n":0.30000001192092896}`)) })
	if r.fatal || len(r.errors) != 1 || !strings.Contains(r.errors[0], "does not round-trip") {
		t.Errorf("RoundTrip did not report the changed float32 spelling: %v", r.errors)
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

type fixedNameCodec struct {
	Known *string `json:"x_jsontest_unknown,omitzero"`
}

func (v *fixedNameCodec) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	type wire fixedNameCodec
	return json.UnmarshalDecode(dec, (*wire)(v))
}

func TestUnknownMemberChecksRejectFixedNameCodec(t *testing.T) {
	tests := map[string]struct{ data string }{
		"error: modeled member absent":  {`{}`},
		"error: modeled member present": {`{"x_jsontest_unknown":"recorded"}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := []byte(tt.data)
			r := run(t, func(tb testing.TB) {
				CheckFallbacks[fixedNameCodec](tb)
				NoUnknown[fixedNameCodec](tb, data, nil)
				InjectUnknown[fixedNameCodec](tb, data)
			})
			if len(r.errors) == 0 || !strings.Contains(r.errors[0], "dropped the unknown member") || !strings.Contains(r.errors[0], injectedName) {
				t.Errorf("checks did not report the dropped probe: %v", r.errors)
			}
		})
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
		"error: sibling embeddings share a fallback": {
			check: func(tb testing.TB) { CheckFallbacks[diamond](tb) },
			want:  []string{"has more than one fallback"},
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

// opaque is a sum type that keeps its branches in unexported fields behind its own JSON methods, the way
// llama.Prompt and openai.MessageContent do.
type opaque struct {
	object *goodInner
	list   []goodInner
}

func (o opaque) MarshalJSONTo(enc *jsontext.Encoder) error {
	if o.object != nil {
		return json.MarshalEncode(enc, o.object)
	}
	return json.MarshalEncode(enc, o.list)
}

func (o *opaque) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if dec.PeekKind() == '{' {
		o.object = new(goodInner)
		return json.UnmarshalDecode(dec, o.object)
	}
	return json.UnmarshalDecode(dec, &o.list)
}

type holdsOpaque struct {
	V     opaque                    `json:"v"`
	Extra map[string]jsontext.Value `json:",embed"`
}

func TestNoUnknownSeesInsideOpaqueTypes(t *testing.T) {
	tests := map[string]struct {
		check func(tb testing.TB)
		want  string
	}{
		"error: record codec owns its fallback": {
			check: func(tb testing.TB) { NoUnknown[opaqueRecord](tb, []byte(`{"future":1}`), nil) },
			want:  "/: future",
		},
		"error: unknown member in the object branch": {
			check: func(tb testing.TB) { NoUnknown[opaque](tb, []byte(`{"b":"x","future":1}`), nil) },
			want:  "/: future",
		},
		"error: unknown member in an array element": {
			check: func(tb testing.TB) { NoUnknown[holdsOpaque](tb, []byte(`{"v":[{"b":"x"},{"b":"y","later":2}]}`), nil) },
			want:  "/v/1: later",
		},
		"success: nothing unknown": {
			check: func(tb testing.TB) { NoUnknown[holdsOpaque](tb, []byte(`{"v":{"b":"x"}}`), nil) },
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			r := run(t, tt.check)
			if tt.want == "" {
				if len(r.errors) != 0 {
					t.Errorf("unexpected failures: %v", r.errors)
				}
				return
			}
			if len(r.errors) != 1 || !strings.Contains(r.errors[0], tt.want) {
				t.Errorf("failures = %v, want one naming %q", r.errors, tt.want)
			}
		})
	}
}

type lossyNumber struct {
	N     float64                   `json:"n"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// 9007199254740993 is 2^53+1, the first integer a float64 cannot hold.
func TestComparisonsKeepNumbersExact(t *testing.T) {
	tests := map[string]struct {
		check    func(tb testing.TB)
		wantFail bool
	}{
		"error: RoundTrip through float64": {
			check:    func(tb testing.TB) { RoundTrip[float64](tb, []byte(`9007199254740993`)) },
			wantFail: true,
		},
		"error: InjectUnknown through a float64 field": {
			check:    func(tb testing.TB) { InjectUnknown[lossyNumber](tb, []byte(`{"n":9007199254740993}`)) },
			wantFail: true,
		},
		"success: large number kept raw in Extra": {
			check: func(tb testing.TB) { InjectUnknown[lossyNumber](tb, []byte(`{"n":1,"big":9007199254740993}`)) },
		},
		"success: spellings of the same value": {
			check: func(tb testing.TB) { RoundTrip[float64](tb, []byte(`1.50e0`)) },
		},
		"success: exact integer through int64": {
			check: func(tb testing.TB) { RoundTrip[int64](tb, []byte(`9007199254740993`)) },
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			r := run(t, tt.check)
			if failed := len(r.errors) > 0; failed != tt.wantFail {
				t.Errorf("failed = %t, want %t: %v", failed, tt.wantFail, r.errors)
			}
		})
	}
}

func TestNumberEqual(t *testing.T) {
	overflow := strings.Repeat("9", 309)
	tests := map[string]struct {
		a, b number
		want bool
	}{
		"success: timing double":                      {"5988.0239520958085", "5988.023952095808", true},
		"success: same decimal double":                {"0.1", "0.10000000000000001", true},
		"success: negative timing double":             {"-5988.0239520958085", "-5988.023952095808", true},
		"success: same subnormal double":              {"5e-324", "5.1e-324", true},
		"success: same integer with decimal":          {"1", "1.0", true},
		"success: same integer with exponent":         {"1.0", "1e0", true},
		"success: equal underflowing rationals":       {"1.5e-400", "15e-401", true},
		"error: adjacent large integers":              {"9007199254740993", "9007199254740992", false},
		"error: adjacent integers with decimal":       {"9007199254740993.0", "9007199254740992.0", false},
		"error: fraction rounds to integer":           {"1.00000000000000001", "1", false},
		"error: overflowing integers":                 {"1e400", "2e400", false},
		"error: overflowing integers with decimal":    {"1.5e400", "1.6e400", false},
		"error: overflowing non-integers":             {number(overflow + ".1"), number(overflow + ".2"), false},
		"error: different doubles":                    {"0.1", "0.1000000000000001", false},
		"error: integer double spelled two ways":      {"9.999999999999999e+22", "1e+23", false},
		"error: distinct fractions underflow to zero": {"1.5e-400", "1.6e-400", false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := tt.a.Equal(tt.b); got != tt.want {
				t.Errorf("%q.Equal(%q) = %t, want %t", tt.a, tt.b, got, tt.want)
			}
			if got := tt.b.Equal(tt.a); got != tt.want {
				t.Errorf("%q.Equal(%q) = %t, want %t", tt.b, tt.a, got, tt.want)
			}
		})
	}
}

// Node embeds itself through a pointer; encoding/json/v2 accepts it.
type Node struct {
	*Node
	A     int                       `json:"a"`
	Extra map[string]jsontext.Value `json:",embed"`
}

func TestCheckFallbacksSelfEmbedding(t *testing.T) {
	done := make(chan *result, 1)
	go func() { done <- run(t, func(tb testing.TB) { CheckFallbacks[Node](tb) }) }()
	select {
	case r := <-done:
		if len(r.errors) != 0 {
			t.Errorf("unexpected failures: %v", r.errors)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("CheckFallbacks did not return for a self-embedding struct")
	}
}
