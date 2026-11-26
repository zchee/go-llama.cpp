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
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func rerankCheckWire[T any](t *testing.T, data []byte) {
	t.Helper()
	jsontest.RoundTrip[T](t, data)
	jsontest.InjectUnknown[T](t, data)
	jsontest.NoUnknown[T](t, data, nil)
	jsontest.CheckFallbacks[T](t)
}

func TestRerankFixtures(t *testing.T) {
	tests := map[string]struct {
		prefix            string
		request, response func(*testing.T, []byte)
	}{
		"success: Jina envelope and token counts":  {"jina", rerankCheckWire[RerankRequest], rerankCheckWire[RerankResponse]},
		"success: TEI array with text":             {"tei", rerankCheckWire[RerankTEIRequest], rerankCheckWire[[]RerankTEIResult]},
		"success: TEI array without text":          {"tei_scores", rerankCheckWire[RerankTEIRequest], rerankCheckWire[[]RerankTEIResult]},
		"error: image input on a text-only model":  {"image", rerankCheckWire[RerankRequest], rerankCheckWire[errorEnvelope]},
		"error: server not configured for ranking": {"disabled", rerankCheckWire[RerankRequest], rerankCheckWire[errorEnvelope]},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			base := "testdata/reranking/" + tt.prefix
			tt.request(t, jsontest.ReadFile(t, base+".req.json"))
			tt.response(t, jsontest.ReadFile(t, base+".resp.json"))
		})
	}
	jsontest.CheckFallbacks[RerankObject](t)
}

func TestRerankNewRequest(t *testing.T) {
	c, err := New("http://localhost/prefix")
	if err != nil {
		t.Fatal(err)
	}
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/k1sAAAAASUVORK5CYII="
	tests := map[string]struct {
		body    any
		fixture string
	}{
		"success: Jina text and image variants": {RerankRequest{
			Query:     RerankObjectInput(RerankObject{Text: StringValue("a pixel"), Image: StringValue("data:image/png;base64," + png)}),
			Documents: []RerankInput{RerankObjectInput(RerankObject{Text: StringSlice("a", " pixel"), Image: StringSlice(png)})},
		}, "image"},
		"success: TEI texts with explicit false": {RerankTEIRequest{Query: RerankText("food"), Texts: []RerankInput{RerankText("bread")}, ReturnText: new(false)}, "tei_scores"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/reranking", tt.body, WithAutoload(false))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != http.MethodPost || req.URL.Path != "/prefix/reranking" || req.URL.RawQuery != "autoload=false" {
				t.Errorf("unexpected request: %s %s", req.Method, req.URL)
			}
			got, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			want := jsontest.ReadFile(t, "testdata/reranking/"+tt.fixture+".req.json")
			if diff := gocmp.Diff(string(want), string(got)); diff != "" {
				t.Errorf("request body (-recorded +built):\n%s", diff)
			}
		})
	}
}

func TestRerankInput(t *testing.T) {
	tests := map[string]struct {
		wire    string
		want    RerankInput
		invalid bool
	}{
		"success: empty string":           {wire: `""`, want: RerankText("")},
		"success: text":                   {wire: `"one"`, want: RerankText("one")},
		"success: empty object":           {wire: `{}`, want: RerankObjectInput(RerankObject{})},
		"success: image only":             {wire: `{"image":["one","two"]}`, want: RerankObjectInput(RerankObject{Image: StringSlice("one", "two")})},
		"error: null input":               {wire: `null`, invalid: true},
		"error: array instead of object":  {wire: `[]`, invalid: true},
		"error: numeric text":             {wire: `{"text":1}`, invalid: true},
		"error: non-string image element": {wire: `{"image":[{}]}`, invalid: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var got RerankInput
			err := json.Unmarshal([]byte(tt.wire), &got)
			if tt.invalid {
				if err == nil {
					t.Fatalf("accepted %s", tt.wire)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if s, ok := got.Text(); ok != (tt.want.Object() == nil) || (ok && s != tt.want.text) {
				t.Errorf("Text() = %q, %t", s, ok)
			}
			if diff := gocmp.Diff(tt.want.Object(), got.Object()); diff != "" {
				t.Errorf("Object() (-want +got):\n%s", diff)
			}
		})
	}
}

func FuzzRerankInput(f *testing.F) {
	for _, name := range []string{"jina", "image", "tei"} {
		var members struct {
			Query     jsontext.Value   `json:"query"`
			Documents []jsontext.Value `json:"documents"`
			Texts     []jsontext.Value `json:"texts"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/reranking/"+name+".req.json"), &members); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(members.Query))
		for _, candidates := range [][]jsontext.Value{members.Documents, members.Texts} {
			for _, v := range candidates {
				f.Add([]byte(v))
			}
		}
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		var v, back RerankInput
		if err := json.Unmarshal(in, &v); err != nil {
			return
		}
		out, err := json.Marshal(v, jsonOptions)
		if err != nil {
			t.Fatalf("encode decoded input: %v", err)
		}
		if err := json.Unmarshal(out, &back); err != nil {
			t.Fatalf("decode %s: %v", out, err)
		}
		// Unknown JSON members may change whitespace or string escaping when re-encoded.
		normalize := gocmp.Transformer("NormalizeJSON", func(raw jsontext.Value) string {
			b, err := json.Marshal(raw, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			return string(b)
		})
		if diff := gocmp.Diff(v, back, gocmp.AllowUnexported(RerankInput{}), normalize); diff != "" {
			t.Fatalf("round trip (-first +second):\n%s", diff)
		}
	})
}

func ExampleClient_Rerank() {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		panic(err)
	}
	res, err := c.Rerank(context.Background(), &RerankRequest{
		Query:     RerankText("Which city is the capital of France?"),
		Documents: []RerankInput{RerankText("Paris is the capital of France."), RerankText("Apples are a fruit.")},
		TopN:      new(1),
	})
	if err != nil {
		panic(err)
	}
	for _, r := range res.Results {
		fmt.Println(r.Index, r.RelevanceScore)
	}
}
