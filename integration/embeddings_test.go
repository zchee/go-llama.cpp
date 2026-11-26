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

package integration

import (
	"errors"
	"math"
	"net/http"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/openai"
)

var presetEmbeddingPooled = &serverPreset{
	name: "embeddings-pooled",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "bert-bge-small"),
			Args:  []string{"--embedding", "--alias", "bert-bge-small", "--n-gpu-layers", "0", "--threads", "2"},
		}
	},
}

var presetEmbeddingPerToken = &serverPreset{
	name: "embeddings-per-token",
	options: func(t testing.TB) harness.Options {
		opts := presetEmbeddingPooled.options(t)
		opts.Args = append(opts.Args, "--pooling", "none")
		return opts
	},
}

func TestEmbeddingNativeAliases(t *testing.T) {
	c, observer := presetEmbeddingPooled.server(t).Client(t)
	want, err := c.Embeddings(t.Context(), &llama.EmbeddingRequest{Input: llama.EmbeddingText("hello"), Normalize: new(2)})
	if err != nil {
		t.Fatal(err)
	}
	if len(want) != 1 || len(want[0].Embedding) != 1 || want[0].Embedding[0].Len() != 384 {
		t.Fatalf("expected one pooled embedding of dimension 384, got %d results", len(want))
	}
	values := want[0].Embedding[0].Values()
	var squares float64
	for _, x := range values {
		squares += x * x
	}
	if norm := math.Sqrt(squares); math.Abs(norm-1) > 1e-5 {
		t.Errorf("pooled vector norm = %g, want 1", norm)
	}

	tests := map[string]struct {
		path string
		req  *llama.EmbeddingRequest
	}{
		"success: legacy alias with input": {
			path: "/embedding",
			req:  &llama.EmbeddingRequest{Input: llama.EmbeddingText("hello"), Normalize: new(2)},
		},
		"success: input takes precedence": {
			path: "/embeddings",
			req:  &llama.EmbeddingRequest{Input: llama.EmbeddingText("hello"), Content: llama.EmbeddingText("world"), Normalize: new(2)},
		},
		"success: OpenAI path returns native for legacy content": {
			path: "/v1/embeddings",
			req:  &llama.EmbeddingRequest{Content: llama.EmbeddingText("hello"), Normalize: new(2), EncodingFormat: new(llama.EmbeddingEncodingBase64)},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := llama.Call[llama.EmbeddingResponse](t.Context(), c, http.MethodPost, test.path, test.req, llama.WithAutoload(false))
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 || got[0].Index != 0 || len(got[0].Embedding) != 1 || got[0].Embedding[0].Format() != llama.EmbeddingEncodingFloat {
				t.Fatalf("expected one native result with one float array, got %d results", len(got))
			}
			if diff := gocmp.Diff(values, got[0].Embedding[0].Values()); diff != "" {
				t.Errorf("embedding vector (-want +got):\n%s", diff)
			}
			exchanges := observer.Exchanges()
			ex := exchanges[len(exchanges)-1]
			if ex.Method != http.MethodPost || ex.Path != test.path || ex.Query.Get("autoload") != "false" || !ex.BodyClosed() {
				t.Errorf("exchange method=%s path=%s query=%v body_closed=%v", ex.Method, ex.Path, ex.Query, ex.BodyClosed())
			}
		})
	}
}

func TestEmbeddingErrors(t *testing.T) {
	pooled, _ := presetEmbeddingPooled.server(t).Client(t)
	perToken, _ := presetEmbeddingPerToken.server(t).Client(t)
	tests := map[string]struct {
		call func(*testing.T) error
		code int
		typ  llama.ErrorType
		msg  string
	}{
		"error: native empty batch": {
			call: func(t *testing.T) error {
				_, err := pooled.Embeddings(t.Context(), &llama.EmbeddingRequest{Input: llama.EmbeddingBatch()})
				return err
			},
			code: http.StatusBadRequest, typ: llama.ErrorTypeInvalidRequest, msg: "must not be empty",
		},
		"error: native unknown encoding": {
			call: func(t *testing.T) error {
				_, err := pooled.Embeddings(t.Context(), &llama.EmbeddingRequest{
					Input: llama.EmbeddingText("hello"), EncodingFormat: new(llama.EmbeddingEncodingFormat("invalid")),
				})
				return err
			},
			code: http.StatusBadRequest, typ: llama.ErrorTypeInvalidRequest, msg: "either float or base64",
		},
		"error: OpenAI pooling none": {
			call: func(t *testing.T) error {
				_, err := openai.NewClient(perToken).Embeddings(t.Context(), &openai.EmbeddingRequest{Input: llama.EmbeddingText("hello")})
				return err
			},
			code: http.StatusBadRequest, typ: llama.ErrorTypeInvalidRequest, msg: "Pooling type 'none'",
		},
		"error: OpenAI media on text-only model": {
			call: func(t *testing.T) error {
				_, err := openai.NewClient(pooled).Embeddings(t.Context(), &openai.EmbeddingRequest{Input: llama.EmbeddingMultimodal(
					llama.ImageURLContentPart("data:image/png;base64,AA=="), llama.TextContentPart("hello"),
				)}, llama.WithNoRetry())
				return err
			},
			code: http.StatusInternalServerError, typ: llama.ErrorTypeServer, msg: "image input is not supported",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			err := test.call(t)
			apiErr, ok := errors.AsType[*llama.APIError](err)
			if !ok || apiErr.StatusCode != test.code || apiErr.Code != test.code || apiErr.Type != test.typ || !strings.Contains(apiErr.Message, test.msg) {
				t.Errorf("got %v, want %d %s with message containing %q", err, test.code, test.typ, test.msg)
			}
		})
	}
}
