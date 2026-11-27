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
	json "encoding/json/v2"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestEmbeddingInputAccessors(t *testing.T) {
	decodeRequest := func(t *testing.T, path string, pick func(*EmbeddingRequest) EmbeddingInput) EmbeddingInput {
		t.Helper()
		var req EmbeddingRequest
		if err := json.Unmarshal(jsontest.ReadFile(t, path), &req, jsonOptions); err != nil {
			t.Fatalf("decoding %s: %v", path, err)
		}
		return pick(&req)
	}

	decodeInputField := func(t *testing.T, path string) EmbeddingInput {
		t.Helper()
		var payload struct {
			Input EmbeddingInput `json:"input"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(t, path), &payload, jsonOptions); err != nil {
			t.Fatalf("decoding %s: %v", path, err)
		}
		return payload.Input
	}

	tests := map[string]struct {
		built       EmbeddingInput
		decoded     func(*testing.T) EmbeddingInput
		wantZero    bool
		wantPrompt  Prompt
		wantObject  *EmbeddingPromptObject
		wantContent *EmbeddingContent
		wantBatch   []EmbeddingInput
	}{
		"success: zero value": {
			built:    EmbeddingInput{},
			wantZero: true,
		},
		"success: text": {
			built: EmbeddingText("hello"),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embedding/legacy.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Content })
			},
			wantPrompt: TextPrompt("hello"),
		},
		"success: tokens": {
			built: EmbeddingTokens(101, 7592, 102),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embeddings/per_token.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Input })
			},
			wantPrompt: TokenPrompt(101, 7592, 102),
		},
		"success: mixed": {
			built: EmbeddingMixed(TokenPart(101), TokenPart(7592), TextPart("world"), TokenPart(102)),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embeddings/mixed_decimal.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Input })
			},
			wantPrompt: MixedPrompt(TokenPart(101), TokenPart(7592), TextPart("world"), TokenPart(102)),
		},
		"success: object": {
			built: EmbeddingObject(EmbeddingPromptObject{PromptString: TokenizeText("<__media__>"), MultimodalData: [][]byte{{0}}}),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embedding/multimodal_refused.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Content })
			},
			wantObject: &EmbeddingPromptObject{PromptString: TokenizeText("<__media__>"), MultimodalData: [][]byte{{0}}},
		},
		"success: content": {
			built: EmbeddingMultimodal(ImageURLContentPart("data:image/png;base64,AA=="), TextContentPart("hello")),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeInputField(t, "openai/testdata/embeddings/multimodal_refused.req.json")
			},
			wantContent: &EmbeddingContent{
				Content: []ContentPart{
					ImageURLContentPart("data:image/png;base64,AA=="),
					TextContentPart("hello"),
				},
			},
		},
		"success: text batch": {
			built: EmbeddingTexts("hello", "world"),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embeddings/pooled.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Input })
			},
			wantBatch: []EmbeddingInput{
				EmbeddingText("hello"),
				EmbeddingText("world"),
			},
		},
		"success: mixed batch": {
			built: EmbeddingBatch(
				EmbeddingTokens(101, 7592, 102),
				EmbeddingMixed(TokenPart(101), TextPart("world"), TokenPart(102)),
				EmbeddingObject(EmbeddingPromptObject{PromptString: TokenizeParts(TextPart("hello"), TextPart("world"))}),
			),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embeddings/batch.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Input })
			},
			wantBatch: []EmbeddingInput{
				EmbeddingTokens(101, 7592, 102),
				EmbeddingMixed(TokenPart(101), TextPart("world"), TokenPart(102)),
				EmbeddingObject(EmbeddingPromptObject{PromptString: TokenizeParts(TextPart("hello"), TextPart("world"))}),
			},
		},
		"success: empty batch": {
			built: EmbeddingBatch(),
			decoded: func(t *testing.T) EmbeddingInput {
				return decodeRequest(t, "testdata/embeddings/empty.req.json", func(r *EmbeddingRequest) EmbeddingInput { return r.Input })
			},
			wantBatch: []EmbeddingInput{},
		},
	}

	cmpOpts := gocmp.Options{
		gocmp.AllowUnexported(EmbeddingInput{}, Prompt{}, PromptPart{}, TokenizeContent{}),
		gocmp.FilterPath(func(p gocmp.Path) bool {
			return p.Last().String() == ".Extra"
		}, cmpopts.EquateEmpty()),
	}

	assertAccessors := func(t *testing.T, label string, got EmbeddingInput, wantZero bool, wantPrompt Prompt, wantObject *EmbeddingPromptObject, wantContent *EmbeddingContent, wantBatch []EmbeddingInput) {
		t.Helper()
		if got.IsZero() != wantZero {
			t.Errorf("%s: IsZero() = %t, want %t", label, got.IsZero(), wantZero)
		}
		if diff := gocmp.Diff(wantPrompt, got.Prompt(), cmpOpts); diff != "" {
			t.Errorf("%s: Prompt() mismatch (-want +got):\n%s", label, diff)
		}
		if diff := gocmp.Diff(wantObject, got.Object(), cmpOpts); diff != "" {
			t.Errorf("%s: Object() mismatch (-want +got):\n%s", label, diff)
		}
		if diff := gocmp.Diff(wantContent, got.Content(), cmpOpts); diff != "" {
			t.Errorf("%s: Content() mismatch (-want +got):\n%s", label, diff)
		}
		if diff := gocmp.Diff(wantBatch, got.Batch(), cmpOpts); diff != "" {
			t.Errorf("%s: Batch() mismatch (-want +got):\n%s", label, diff)
		}
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			assertAccessors(t, "built", tt.built, tt.wantZero, tt.wantPrompt, tt.wantObject, tt.wantContent, tt.wantBatch)
			if tt.decoded != nil {
				assertAccessors(t, "decoded", tt.decoded(t), tt.wantZero, tt.wantPrompt, tt.wantObject, tt.wantContent, tt.wantBatch)
			}
		})
	}
}
