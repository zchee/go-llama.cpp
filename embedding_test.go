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

package llama

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

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestEmbeddingWire(t *testing.T) {
	jsontest.CheckFallbacks[EmbeddingRequest](t)
	jsontest.CheckFallbacks[EmbeddingResponse](t)
	jsontest.CheckFallbacks[EmbeddingPromptObject](t)
	jsontest.CheckFallbacks[EmbeddingContent](t)
	jsontest.CheckFallbacks[errorEnvelope](t)

	tests := map[string]struct {
		fixture string
		vectors []int
		code    int
		typ     ErrorType
	}{
		"success: legacy":              {fixture: "embedding/legacy", vectors: []int{1}},
		"success: pooled":              {fixture: "embeddings/pooled", vectors: []int{1, 1}},
		"success: per token":           {fixture: "embeddings/per_token", vectors: []int{3}},
		"success: token batch":         {fixture: "embeddings/batch", vectors: []int{1, 1, 1}},
		"success: OpenAI fallback":     {fixture: "embeddings/openai_legacy", vectors: []int{1}},
		"success: input wins":          {fixture: "embeddings/input_precedence", vectors: []int{1}},
		"success: mixed decimal token": {fixture: "embeddings/mixed_decimal", vectors: []int{1}},
		"error: empty batch":           {fixture: "embeddings/empty", code: 400, typ: ErrorTypeInvalidRequest},
		"error: encoding":              {fixture: "embeddings/invalid_encoding", code: 400, typ: ErrorTypeInvalidRequest},
		"error: legacy media":          {fixture: "embedding/multimodal_refused", code: 500, typ: ErrorTypeServer},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			in := jsontest.ReadFile(t, "testdata/"+test.fixture+".req.json")
			jsontest.RoundTrip[EmbeddingRequest](t, in)
			jsontest.InjectUnknown[EmbeddingRequest](t, in)
			jsontest.NoUnknown[EmbeddingRequest](t, in, nil)
			out := jsontest.ReadFile(t, "testdata/"+test.fixture+".resp.json")
			if test.code != 0 {
				jsontest.RoundTrip[errorEnvelope](t, out)
				jsontest.InjectUnknown[errorEnvelope](t, out)
				jsontest.NoUnknown[errorEnvelope](t, out, nil)
				got := ReadAPIError(&http.Response{StatusCode: test.code, Body: io.NopCloser(bytes.NewReader(out))})
				if got.Code != test.code || got.Type != test.typ || got.Message == "" {
					t.Errorf("ReadAPIError = %v, want code %d and type %s with a message", got, test.code, test.typ)
				}
				return
			}
			got := *jsontest.RoundTrip[EmbeddingResponse](t, out)
			jsontest.InjectUnknown[EmbeddingResponse](t, out)
			jsontest.NoUnknown[EmbeddingResponse](t, out, nil)
			var lengths []int
			for i, item := range got {
				if item.Index != i {
					t.Errorf("result[%d].Index = %d", i, item.Index)
				}
				lengths = append(lengths, len(item.Embedding))
				for j, vector := range item.Embedding {
					if vector.Len() != 384 || vector.Format() != EmbeddingEncodingFloat {
						t.Errorf("result[%d].Embedding[%d]: dimensions=%d, format=%q; want 384 and float", i, j, vector.Len(), vector.Format())
					}
				}
			}
			if diff := gocmp.Diff(test.vectors, lengths); diff != "" {
				t.Errorf("number of vectors per result (-want +got):\n%s", diff)
			}
		})
	}
}

func TestEmbeddingNewRequest(t *testing.T) {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		path, wantPath string
		fixture        string
		body           *EmbeddingRequest
	}{
		"success: legacy text": {
			path: embeddingPath, wantPath: "/embedding", fixture: "embedding/legacy",
			body: &EmbeddingRequest{Content: EmbeddingText("hello"), Normalize: new(0)},
		},
		"success: text list": {
			path: embeddingsPath, wantPath: "/embeddings", fixture: "embeddings/pooled",
			body: &EmbeddingRequest{Input: EmbeddingTexts("hello", "world"), EncodingFormat: new(EmbeddingEncodingBase64)},
		},
		"success: token ids": {
			path: embeddingsPath, wantPath: "/embeddings", fixture: "embeddings/per_token",
			body: &EmbeddingRequest{Input: EmbeddingTokens(101, 7592, 102), Normalize: new(2)},
		},
		"success: mixed batch": {
			path: embeddingsPath, wantPath: "/embeddings", fixture: "embeddings/batch",
			body: &EmbeddingRequest{Input: EmbeddingBatch(
				EmbeddingTokens(101, 7592, 102),
				EmbeddingMixed(TokenPart(101), TextPart("world"), TokenPart(102)),
				EmbeddingObject(EmbeddingPromptObject{PromptString: TokenizeParts(TextPart("hello"), TextPart("world"))}),
			)},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, test.path, test.body, WithAutoload(false))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != http.MethodPost || req.URL.Path != test.wantPath || req.URL.RawQuery != "autoload=false" {
				t.Errorf("request = %s %s, want POST %s?autoload=false", req.Method, req.URL, test.wantPath)
			}
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			want := jsontext.Value(jsontest.ReadFile(t, "testdata/"+test.fixture+".req.json"))
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

func TestEmbeddingRequestRejectsInvalidInput(t *testing.T) {
	tests := map[string]struct {
		fixture string
		body    string
	}{
		"error: null input takes precedence":        {fixture: "null_input"},
		"error: null legacy content":                {body: `{"content":null}`},
		"error: null media selects multimodal":      {fixture: "null_media"},
		"error: nested prompt":                      {fixture: "nested_prompt"},
		"error: floating point token list":          {fixture: "float_only"},
		"error: no integer spelling in mixed input": {body: `{"input":[7592.0,"world"]}`},
		"error: mixed numeric list without text":    {body: `{"input":[101,7592.0]}`},
		"error: non-integral mixed token":           {body: `{"input":[101,7592.0000000000000001,"world"]}`},
		"error: out of range mixed token":           {body: `{"input":[101,2147483648.0,"world"]}`},
		"error: array prompt with media":            {body: `{"input":{"prompt_string":[101],"multimodal_data":[]}}`},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			data := []byte(test.body)
			if test.fixture != "" {
				data = jsontest.ReadFile(t, "testdata/embeddings/"+test.fixture+".req.json")
			}
			if err := json.Unmarshal(data, new(EmbeddingRequest)); err == nil {
				t.Errorf("decoded invalid input: %s", data)
			}
		})
	}
}

func TestEmbeddingPromptObjectRejectsInvalidShape(t *testing.T) {
	tests := map[string]struct{ object EmbeddingPromptObject }{
		"error: unset prompt":        {EmbeddingPromptObject{}},
		"error: media requires text": {EmbeddingPromptObject{PromptString: TokenizeParts(TokenPart(101)), MultimodalData: [][]byte{}}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if got, err := json.Marshal(EmbeddingObject(test.object)); err == nil {
				t.Errorf("encoded invalid prompt: %s", got)
			}
		})
	}
}

func TestEmbeddingVectorBase64Precision(t *testing.T) {
	v := EmbeddingVectorOf(EmbeddingEncodingBase64, 0.1)
	if diff := gocmp.Diff([]float64{float64(float32(0.1))}, v.Values()); diff != "" {
		t.Errorf("base64 values before encoding (-want +got):\n%s", diff)
	}
}

func TestEmbeddingVectorOf(t *testing.T) {
	tests := map[string]struct {
		format EmbeddingEncodingFormat
		json   string
	}{
		"success: default float": {json: `[1,-2]`},
		"success: base64":        {format: EmbeddingEncodingBase64, json: `"AACAPwAAAMA="`},
		"error: unknown format":  {format: "unknown"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			vals := []float64{1, -2}
			v := EmbeddingVectorOf(test.format, vals...)
			vals[0] = 7
			v.Values()[1] = 7
			got, err := json.Marshal(v)
			if test.json == "" {
				if err == nil {
					t.Errorf("Marshal succeeded: %s", got)
				}
				return
			}
			if err != nil || string(got) != test.json {
				t.Errorf("Marshal = %s, %v; want %s", got, err, test.json)
			}
		})
	}
}

func FuzzEmbeddingInput(f *testing.F) {
	for _, seed := range []string{
		`[{"content":[{"type":""}]}]`,
		`{"content":[{"type":"text","text":""},{"type":"image_url","image_url":{"url":null}}]}`,
		`{"content":[{"type":"text","text":"","x-app":null}]}`,
		`{"prompt_string":"hello","multimodal_data":[null]}`,
		`{"prompt_string":"hello","multimodal_data":["AB=="]}`,
	} {
		f.Add([]byte(seed))
	}
	for _, path := range []string{
		"testdata/embedding/legacy.req.json",
		"testdata/embedding/multimodal_refused.req.json",
		"testdata/embeddings/pooled.req.json",
		"testdata/embeddings/per_token.req.json",
		"testdata/embeddings/batch.req.json",
		"testdata/embeddings/null_media.req.json",
		"testdata/embeddings/mixed_decimal.req.json",
		"openai/testdata/embeddings/content.req.json",
		"openai/testdata/embeddings/multimodal_refused.req.json",
	} {
		var fields map[string]jsontext.Value
		if err := json.Unmarshal(jsontest.ReadFile(f, path), &fields); err != nil {
			f.Fatal(err)
		}
		for _, key := range []string{"input", "content"} {
			if v, ok := fields[key]; ok {
				f.Add([]byte(v))
			}
		}
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var in EmbeddingInput
		if err := json.Unmarshal(data, &in); err != nil {
			return
		}
		jsontest.RoundTrip[EmbeddingInput](t, data)
		first, err := json.Marshal(in, jsonOptions)
		if err != nil {
			t.Fatal(err)
		}
		var again EmbeddingInput
		if err := json.Unmarshal(first, &again); err != nil {
			t.Fatalf("decoding re-encoded embedding input %s: %v", first, err)
		}
		// Compare the JSON values since the codec can normalize formatting within preserved unknown members.
		second, err := json.Marshal(again, jsonOptions)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(first, second) {
			t.Errorf("embedding input changed after decoding again: %s != %s", first, second)
		}
	})
}

func ExampleClient_Embedding() {
	client, err := New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	results, err := client.Embedding(context.Background(), &EmbeddingRequest{Content: EmbeddingText("hello")})
	if err != nil {
		panic(err)
	}
	fmt.Println(results[0].Embedding[0].Len())
}

func ExampleClient_Embeddings() {
	client, err := New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	results, err := client.Embeddings(context.Background(), &EmbeddingRequest{Input: EmbeddingTexts("hello", "world")})
	if err != nil {
		panic(err)
	}
	fmt.Println(len(results))
}
