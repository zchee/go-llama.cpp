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

package systemone_test

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/systemone"
)

func TestSystemoneWire(t *testing.T) {
	tests := map[string]struct {
		name   string
		status int
		kind   llama.ErrorType
	}{
		"success: all question types":     {name: "mixed"},
		"success: image alias":            {name: "image"},
		"success: files with array state": {name: "files"},
		"success: numeric state":          {name: "number"},
		"success: boolean state":          {name: "boolean"},
		"error: null state":               {name: "invalid", status: 400, kind: llama.ErrorTypeInvalidRequest},
		"error: nondecision model":        {name: "unsupported", status: 501, kind: llama.ErrorTypeNotSupported},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/systemone/" + tt.name
			req := jsontest.ReadFile(t, prefix+".req.json")
			jsontest.RoundTrip[systemone.Request](t, req)
			jsontest.NoUnknown[systemone.Request](t, req, nil)
			// Questions are keyed by caller-defined ID, and choice criteria by option label.
			jsontest.InjectUnknown[systemone.Request](t, req, jsontest.SkipObjects("/questions", "/questions/route/criteria"))
			res := jsontest.ReadFile(t, prefix+".resp.json")
			if tt.status != 0 {
				e := llama.ReadAPIError(&http.Response{StatusCode: tt.status, Body: io.NopCloser(bytes.NewReader(res))})
				if e.Code != tt.status || e.Type != tt.kind || e.Message == "" {
					t.Fatalf("ReadAPIError = %+v", e)
				}
				return
			}
			jsontest.RoundTrip[systemone.Response](t, res)
			jsontest.NoUnknown[systemone.Response](t, res, nil)
			// Answers, probabilities and legends are keyed maps, not records.
			jsontest.InjectUnknown[systemone.Response](t, res, jsontest.SkipObjects("/answers", "/answers/*/probabilities", "/answers/*/legend"))
		})
	}
	jsontest.CheckFallbacks[systemone.Request](t)
	jsontest.CheckFallbacks[systemone.Response](t)
	jsontest.CheckFallbacks[systemone.ChoiceAnswer](t)
	jsontest.CheckFallbacks[systemone.ScoreAnswer](t)
	jsontest.CheckFallbacks[systemone.NoulAnswer](t)
}

func TestSystemoneNewRequest(t *testing.T) {
	c, err := llama.New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		req  systemone.Request
		want string
	}{
		"success: explicit null state and nullable descriptions": {req: systemone.Request{State: systemone.Null(), Questions: map[string]systemone.Question{"q": systemone.Choice(systemone.Null(), systemone.ChoiceCriteria{"x": systemone.Null()})}}, want: `{"state":null,"questions":{"q":{"type":"choice","instructions":null,"criteria":{"x":null}}}}`},
		"success: model for router mode":                         {req: systemone.Request{Model: "tinyopenjev", State: systemone.Text("ok"), Questions: map[string]systemone.Question{"q": systemone.Noul(systemone.Payload{}, nil)}}, want: `{"model":"tinyopenjev","state":"ok","questions":{"q":{"type":"noul"}}}`},
		"success: omitted default instructions":                  {req: systemone.Request{State: systemone.Text(""), Questions: map[string]systemone.Question{"q": systemone.Noul(systemone.Payload{}, nil)}, Images: []string{}}, want: `{"state":"","questions":{"q":{"type":"noul"}},"images":[]}`},
		"error: explicit alias conflict":                         {req: systemone.Request{State: systemone.Null(), Files: []string{}, Images: []string{}}},
		"error: alias supplied through Extra":                    {req: systemone.Request{State: systemone.Null(), Files: []string{"a"}, Extra: map[string]jsontext.Value{"images": jsontext.Value(`[]`)}}},
		"error: questions supplied through Extra":                {req: systemone.Request{State: systemone.Null(), Extra: map[string]jsontext.Value{"questions": jsontext.Value(`{}`)}}},
		"error: model supplied through Extra":                    {req: systemone.Request{State: systemone.Null(), Questions: map[string]systemone.Question{"q": systemone.Noul(systemone.Payload{}, nil)}, Extra: map[string]jsontext.Value{"model": jsontext.Value(`"tinyopenjev"`)}}},
		"error: omitted files supplied through Extra":            {req: systemone.Request{State: systemone.Null(), Extra: map[string]jsontext.Value{"files": jsontext.Value(`[]`)}}},
		"error: omitted instructions supplied through Extra":     {req: systemone.Request{State: systemone.Null(), Questions: map[string]systemone.Question{"q": {Type: "noul", Extra: map[string]jsontext.Value{"instructions": jsontext.Value(`null`)}}}}},
		"error: mismatched criteria":                             {req: systemone.Request{State: systemone.Null(), Questions: map[string]systemone.Question{"q": {Type: "score", Criteria: systemone.ChoiceCriteria{}}}}},
		"error: unset state":                                     {},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/v1/systemone", &tt.req, llama.WithQuery("model", "jev"))
			if tt.want == "" {
				if err == nil {
					req.Body.Close()
					t.Fatal("NewRequest accepted conflicting or missing request fields")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != "POST" || req.URL.RequestURI() != "/v1/systemone?model=jev" {
				t.Fatalf("unexpected request: %s %s", req.Method, req.URL)
			}
			data, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tt.want, string(data)); diff != "" {
				t.Errorf("request body (-want +got):\n%s", diff)
			}
		})
	}
}

func ExampleClient_Evaluate() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	res, err := systemone.NewClient(c).Evaluate(context.Background(), &systemone.Request{
		State: systemone.Text("My parcel was delivered."),
		Questions: map[string]systemone.Question{
			"delivered": systemone.Noul(systemone.Text("Has the parcel arrived?"), nil),
		},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Answers["delivered"].Value.(systemone.NoulAnswer).Noul)
}

func TestSystemonePayloadNumberAndBoolean(t *testing.T) {
	tests := map[string]struct {
		raw     string
		val     any
		wantErr bool
	}{
		"success: integer":         {raw: `42`, val: 42},
		"success: exponent number": {raw: `1e5`},
		"success: negative float":  {raw: `-3.50`, val: -3.5},
		"success: zero":            {raw: `0`, val: 0},
		"success: true":            {raw: `true`, val: true},
		"success: false":           {raw: `false`, val: false},
		"success: object order":    {raw: `{"b":1,"a":2}`},
		"success: text":            {raw: `"hello"`, val: "hello"},
		"success: null":            {raw: `null`},
		"error: empty raw":         {raw: ``, wantErr: true},
		"error: malformed raw":     {raw: `{bad`, wantErr: true},
		"error: trailing token":    {raw: `1 2`, wantErr: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			rawPayload, err := systemone.Raw(jsontext.Value(tt.raw))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Raw(%q) succeeded, want error", tt.raw)
				}
				return
			}
			if err != nil {
				t.Fatalf("Raw(%q): %v", tt.raw, err)
			}
			if rawPayload.IsZero() {
				t.Fatalf("Raw(%q).IsZero() = true, want false", tt.raw)
			}
			gotRaw, err := json.Marshal(rawPayload, json.Deterministic(true))
			if err != nil {
				t.Fatalf("Marshal(Raw(%q)): %v", tt.raw, err)
			}
			if string(gotRaw) != tt.raw {
				t.Errorf("Marshal(Raw(%q)) = %s, want %s", tt.raw, gotRaw, tt.raw)
			}

			var p systemone.Payload
			if err := json.Unmarshal([]byte(tt.raw), &p); err != nil {
				t.Fatalf("Unmarshal(%s): %v", tt.raw, err)
			}
			if p.IsZero() {
				t.Fatalf("Payload(%s).IsZero() = true, want false", tt.raw)
			}
			got, err := json.Marshal(p, json.Deterministic(true))
			if err != nil {
				t.Fatalf("Marshal(%s): %v", tt.raw, err)
			}
			if string(got) != tt.raw {
				t.Errorf("Marshal = %s, want %s", got, tt.raw)
			}
			if tt.val != nil {
				np, err := systemone.NewPayload(tt.val)
				if err != nil {
					t.Fatalf("NewPayload(%v): %v", tt.val, err)
				}
				gotNew, err := json.Marshal(np, json.Deterministic(true))
				if err != nil {
					t.Fatalf("Marshal(NewPayload(%v)): %v", tt.val, err)
				}
				if tt.raw != `-3.50` && string(gotNew) != tt.raw {
					t.Errorf("NewPayload(%v) Marshal = %s, want %s", tt.val, gotNew, tt.raw)
				}
			}
		})
	}
}

func TestSystemoneMarshalRefusesExtraModel(t *testing.T) {
	req := systemone.Request{
		State:     systemone.Null(),
		Questions: map[string]systemone.Question{"q": systemone.Noul(systemone.Payload{}, nil)},
		Extra:     map[string]jsontext.Value{"model": jsontext.Value(`"tinyopenjev"`)},
	}
	if out, err := json.Marshal(req); err == nil {
		t.Fatalf("json.Marshal accepted Extra[\"model\"]: %s", out)
	}
}

func systemoneFuzz[T any](f *testing.F) {
	f.Helper()
	f.Fuzz(func(t *testing.T, data []byte) {
		var a, b T
		if err := json.Unmarshal(data, &a); err != nil {
			return
		}
		out, err := json.Marshal(a, json.Deterministic(true))
		if err != nil {
			t.Fatalf("encode after successful decode: %v", err)
		}
		if err := json.Unmarshal(out, &b); err != nil {
			t.Fatalf("decode after encode: %v", err)
		}
		if diff := gocmp.Diff(a, b, gocmp.AllowUnexported(systemone.Payload{}), gocmp.Comparer(func(x, y jsontext.Value) bool {
			if x == nil || y == nil {
				return x == nil && y == nil
			}
			a, err := json.Marshal(x)
			if err != nil {
				return false
			}
			b, err := json.Marshal(y)
			return err == nil && bytes.Equal(a, b)
		})); diff != "" {
			t.Fatalf("unstable decoded value (-first +second):\n%s", diff)
		}
	})
}

func FuzzSystemonePayload(f *testing.F) {
	for _, name := range []string{"mixed", "image", "files", "invalid", "number", "boolean"} {
		var req struct {
			State jsontext.Value `json:"state"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/systemone/"+name+".req.json"), &req); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(req.State))
	}
	systemoneFuzz[systemone.Payload](f)
}

func FuzzSystemoneQuestion(f *testing.F) {
	for _, name := range []string{"mixed", "files", "image", "number", "boolean"} {
		var req struct {
			Questions map[string]jsontext.Value `json:"questions"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/systemone/"+name+".req.json"), &req); err != nil {
			f.Fatal(err)
		}
		for _, q := range req.Questions {
			f.Add([]byte(q))
		}
	}
	f.Add([]byte(`{"type":"noul","criteria":null}`))
	systemoneFuzz[systemone.Question](f)
}

func FuzzSystemoneAnswer(f *testing.F) {
	for _, name := range []string{"mixed", "files", "number", "boolean"} {
		var res struct {
			Answers map[string]jsontext.Value `json:"answers"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/systemone/"+name+".resp.json"), &res); err != nil {
			f.Fatal(err)
		}
		for _, a := range res.Answers {
			f.Add([]byte(a))
		}
	}
	f.Add([]byte(`{"type":"choice"}`))
	systemoneFuzz[systemone.Answer](f)
}
