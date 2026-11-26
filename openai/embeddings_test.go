// Copyright 2026 The go-llama.cpp Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openai

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
)

func TestEmbeddingsWire(t *testing.T) {
	jsontest.CheckFallbacks[EmbeddingRequest](t)
	jsontest.CheckFallbacks[EmbeddingResponse](t)

	tests := map[string]struct {
		fixture string
		model   string
		count   int
		format  llama.EmbeddingEncodingFormat
		code    int
		typ     llama.ErrorType
	}{
		"success: float":   {fixture: "float", model: "caller-alias", count: 1, format: llama.EmbeddingEncodingFloat},
		"success: base64":  {fixture: "base64", model: "caller-alias", count: 1, format: llama.EmbeddingEncodingBase64},
		"success: content": {fixture: "content", model: "bert-bge-small", count: 2, format: llama.EmbeddingEncodingFloat},
		"error: image":     {fixture: "multimodal_refused", code: 500, typ: llama.ErrorTypeServer},
		"error: pooling":   {fixture: "pooling_none_refused", code: 400, typ: llama.ErrorTypeInvalidRequest},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			in := jsontest.ReadFile(t, "testdata/embeddings/"+test.fixture+".req.json")
			jsontest.RoundTrip[EmbeddingRequest](t, in)
			jsontest.InjectUnknown[EmbeddingRequest](t, in)
			jsontest.NoUnknown[EmbeddingRequest](t, in, nil)
			out := jsontest.ReadFile(t, "testdata/embeddings/"+test.fixture+".resp.json")
			if test.code != 0 {
				got := llama.ReadAPIError(&http.Response{StatusCode: test.code, Body: io.NopCloser(bytes.NewReader(out))})
				if got.Code != test.code || got.Type != test.typ || got.Message == "" {
					t.Errorf("ReadAPIError = %v, want code %d and type %s with a message", got, test.code, test.typ)
				}
				return
			}
			got := jsontest.RoundTrip[EmbeddingResponse](t, out)
			jsontest.InjectUnknown[EmbeddingResponse](t, out)
			jsontest.NoUnknown[EmbeddingResponse](t, out, nil)
			if got.Object != "list" || got.Model != test.model || len(got.Data) != test.count {
				t.Errorf("object=%q, model=%q, results=%d; want list, %q, %d", got.Object, got.Model, len(got.Data), test.model, test.count)
			}
			if got.Usage.PromptTokens != 3*test.count || got.Usage.TotalTokens != got.Usage.PromptTokens {
				t.Errorf("usage = %+v, want %d prompt and total tokens", got.Usage, 3*test.count)
			}
			for i, item := range got.Data {
				if item.Index != i || item.Object != "embedding" || item.Embedding.Len() != 384 || item.Embedding.Format() != test.format {
					t.Errorf("data[%d]: index=%d, object=%q, dimensions=%d, format=%q", i, item.Index, item.Object, item.Embedding.Len(), item.Embedding.Format())
				}
				if test.format == llama.EmbeddingEncodingBase64 {
					if item.EncodingFormat == nil || *item.EncodingFormat != test.format {
						t.Errorf("data[%d].EncodingFormat = %v, want base64", i, item.EncodingFormat)
					}
				} else if item.EncodingFormat != nil {
					t.Errorf("data[%d] unexpectedly has encoding_format", i)
				}
			}
		})
	}
}

func TestEmbeddingsNumericValues(t *testing.T) {
	floats := embeddingVectorFixture(t, "float")
	base64 := embeddingVectorFixture(t, "base64")
	if diff := gocmp.Diff(floats.Values(), base64.Values()); diff != "" {
		t.Errorf("numeric values differ (-float +base64):\n%s", diff)
	}
}

func TestEmbeddingsNewRequest(t *testing.T) {
	c, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		fixture string
		body    *EmbeddingRequest
	}{
		"success: float": {
			fixture: "float",
			body:    &EmbeddingRequest{Input: llama.EmbeddingText("hello"), Model: "caller-alias", EncodingFormat: new(llama.EmbeddingEncodingFloat)},
		},
		"success: image and text": {
			fixture: "multimodal_refused",
			body: &EmbeddingRequest{Input: llama.EmbeddingMultimodal(
				llama.ImageURLContentPart("data:image/png;base64,AA=="), llama.TextContentPart("hello"),
			)},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, embeddingsPath, test.body, llama.WithAutoload(false))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != http.MethodPost || req.URL.Path != "/v1/embeddings" || req.URL.RawQuery != "autoload=false" {
				t.Errorf("request = %s %s, want POST /v1/embeddings?autoload=false", req.Method, req.URL)
			}
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			want := jsontext.Value(jsontest.ReadFile(t, "testdata/embeddings/"+test.fixture+".req.json"))
			got := jsontext.Value(body)
			if err := want.Canonicalize(); err != nil {
				t.Fatal(err)
			}
			if err := got.Canonicalize(); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(want, got) {
				t.Errorf("request body = %s, want %s", got, want)
			}
		})
	}
}

func TestEmbeddingVectorInvalid(t *testing.T) {
	tests := map[string]struct{ value string }{
		"error: non-array or string": {value: `false`},
		"error: wrong element type":  {value: `["0"]`},
		"error: null value":          {value: `[null]`},
		"error: malformed base64":    {value: `"!"`},
		"error: partial float32":     {value: `"AA=="`},
		"error: nonfinite float32":   {value: `"AACAfw=="`},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var vector EmbeddingVector
			if err := json.Unmarshal([]byte(test.value), &vector); err == nil {
				t.Errorf("decoded invalid vector %s: %+v", test.value, vector)
			}
		})
	}
}

func embeddingVectorFixture(t testing.TB, name string) EmbeddingVector {
	t.Helper()
	data := jsontest.ReadFile(t, "testdata/embeddings/"+name+".resp.json")
	var res EmbeddingResponse
	if err := json.Unmarshal(data, &res); err != nil {
		t.Fatal(err)
	}
	if len(res.Data) != 1 {
		t.Fatalf("fixture has %d items, want one", len(res.Data))
	}
	return res.Data[0].Embedding
}

func FuzzEmbeddingVector(f *testing.F) {
	for _, name := range []string{"float", "base64"} {
		data, err := json.Marshal(embeddingVectorFixture(f, name))
		if err != nil {
			f.Fatal(err)
		}
		f.Add(data)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var v EmbeddingVector
		if err := json.Unmarshal(data, &v); err != nil {
			return
		}
		out, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		var again EmbeddingVector
		if err := json.Unmarshal(out, &again); err != nil {
			t.Fatal(err)
		}
		if v.Format() != again.Format() {
			t.Errorf("re-encoded format = %q, want %q", again.Format(), v.Format())
		}
		if diff := gocmp.Diff(v.Values(), again.Values()); diff != "" {
			t.Errorf("re-encoded vector changed (-first +second):\n%s", diff)
		}
	})
}

func ExampleClient_Embeddings() {
	client, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	resp, err := NewClient(client).Embeddings(context.Background(), &EmbeddingRequest{
		Input:          llama.EmbeddingText("hello"),
		EncodingFormat: new(llama.EmbeddingEncodingBase64),
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Data[0].Embedding.Len())
}
