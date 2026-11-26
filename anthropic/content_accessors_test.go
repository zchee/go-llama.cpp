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

package anthropic

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestContentAccessors(t *testing.T) {
	decodeFixture := func(t *testing.T, path string, selectContent func(*MessagesRequest) Content) Content {
		t.Helper()
		var req MessagesRequest
		if err := json.Unmarshal(jsontest.ReadFile(t, path), &req); err != nil {
			t.Fatalf("decoding %s: %v", path, err)
		}
		return selectContent(&req)
	}

	decodeJSON := func(t *testing.T, raw string) Content {
		t.Helper()
		var c Content
		if err := json.Unmarshal([]byte(raw), &c); err != nil {
			t.Fatalf("decoding %s: %v", raw, err)
		}
		return c
	}

	tests := map[string]struct {
		built      Content
		decoded    func(*testing.T) Content
		wantZero   bool
		wantText   string
		wantIsText bool
		wantBlocks []ContentBlock
		wantNull   bool
		wantRaw    jsontext.Value
	}{
		"success: zero value": {
			built:    Content{},
			wantZero: true,
		},
		"success: text": {
			built: TextContent("Tell a short story."),
			decoded: func(t *testing.T) Content {
				return decodeFixture(t, "testdata/messages/plain.req.json", func(r *MessagesRequest) Content { return r.System })
			},
			wantText:   "Tell a short story.",
			wantIsText: true,
		},
		"success: empty text": {
			built: TextContent(""),
			decoded: func(t *testing.T) Content {
				return decodeFixture(t, "testdata/count_tokens/empty_result.req.json", func(r *MessagesRequest) Content { return r.Messages[2].Content.Blocks()[0].Content })
			},
			wantText:   "",
			wantIsText: true,
		},
		"success: blocks": {
			built: BlocksContent(ContentBlock{Type: "text", Text: new("Be brief.")}),
			decoded: func(t *testing.T) Content {
				return decodeFixture(t, "testdata/count_tokens/history.req.json", func(r *MessagesRequest) Content { return r.System })
			},
			wantBlocks: []ContentBlock{{Type: "text", Text: new("Be brief.")}},
		},
		"success: empty blocks": {
			built: BlocksContent(),
			decoded: func(t *testing.T) Content {
				return decodeFixture(t, "testdata/count_tokens/references.req.json", func(r *MessagesRequest) Content { return r.System })
			},
			wantBlocks: []ContentBlock{},
		},
		"success: null": {
			built: NullContent(),
			decoded: func(t *testing.T) Content {
				return decodeFixture(t, "testdata/count_tokens/null.req.json", func(r *MessagesRequest) Content { return r.System })
			},
			wantNull: true,
		},
		"success: raw object": {
			built: RawContent(jsontext.Value(`{"a":1}`)),
			decoded: func(t *testing.T) Content {
				return decodeFixture(t, "testdata/count_tokens/raw_result.req.json", func(r *MessagesRequest) Content { return r.Messages[2].Content.Blocks()[0].Content })
			},
			wantRaw: jsontext.Value(`{"a":1}`),
		},
		"success: raw array": {
			built:   RawContent(jsontext.Value(`[2,false,[],null]`)),
			decoded: func(t *testing.T) Content { return decodeJSON(t, `[2,false,[],null]`) },
			wantRaw: jsontext.Value(`[2,false,[],null]`),
		},
		"success: raw number": {
			built:   RawContent(jsontext.Value(`17`)),
			decoded: func(t *testing.T) Content { return decodeJSON(t, `17`) },
			wantRaw: jsontext.Value(`17`),
		},
		"success: raw boolean": {
			built:   RawContent(jsontext.Value(`false`)),
			decoded: func(t *testing.T) Content { return decodeJSON(t, `false`) },
			wantRaw: jsontext.Value(`false`),
		},
	}

	assertAccessors := func(t *testing.T, label string, got Content, wantZero bool, wantText string, wantIsText bool, wantBlocks []ContentBlock, wantNull bool, wantRaw jsontext.Value) {
		t.Helper()
		if got.IsZero() != wantZero {
			t.Errorf("%s: IsZero() = %t, want %t", label, got.IsZero(), wantZero)
		}
		text, isText := got.Text()
		if text != wantText || isText != wantIsText {
			t.Errorf("%s: Text() = (%q, %t), want (%q, %t)", label, text, isText, wantText, wantIsText)
		}
		blocks := got.Blocks()
		if (blocks != nil) != (wantBlocks != nil) {
			t.Errorf("%s: Blocks() nilness = %t, want %t", label, blocks == nil, wantBlocks == nil)
		}
		if diff := gocmp.Diff(wantBlocks, blocks, gocmp.AllowUnexported(Content{}, ContentBlock{})); diff != "" {
			t.Errorf("%s: Blocks() mismatch (-want +got):\n%s", label, diff)
		}
		if got.IsNull() != wantNull {
			t.Errorf("%s: IsNull() = %t, want %t", label, got.IsNull(), wantNull)
		}
		raw := got.Raw()
		if (raw != nil) != (wantRaw != nil) {
			t.Errorf("%s: Raw() nilness = %t, want %t", label, raw == nil, wantRaw == nil)
		}
		if diff := gocmp.Diff(wantRaw, raw); diff != "" {
			t.Errorf("%s: Raw() mismatch (-want +got):\n%s", label, diff)
		}
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			assertAccessors(t, "built", tt.built, tt.wantZero, tt.wantText, tt.wantIsText, tt.wantBlocks, tt.wantNull, tt.wantRaw)
			if tt.decoded != nil {
				assertAccessors(t, "decoded", tt.decoded(t), tt.wantZero, tt.wantText, tt.wantIsText, tt.wantBlocks, tt.wantNull, tt.wantRaw)
			}
		})
	}
}

func TestContentBlockIsNull(t *testing.T) {
	var req MessagesRequest
	if err := json.Unmarshal(jsontest.ReadFile(t, "testdata/count_tokens/raw_result.req.json"), &req); err != nil {
		t.Fatal(err)
	}
	blocks := req.Messages[0].Content.Blocks()
	if len(blocks) != 3 {
		t.Fatalf("len(blocks) = %d, want 3", len(blocks))
	}

	var directNull ContentBlock
	if err := json.Unmarshal([]byte(`null`), &directNull); err != nil {
		t.Fatal(err)
	}

	tests := map[string]struct {
		block    ContentBlock
		wantNull bool
	}{
		"success: zero block":              {block: ContentBlock{}, wantNull: false},
		"success: text block from fixture": {block: blocks[0], wantNull: false},
		"success: untagged block":          {block: blocks[1], wantNull: false},
		"success: null block from fixture": {block: blocks[2], wantNull: true},
		"success: direct null JSON":        {block: directNull, wantNull: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := tt.block.IsNull(); got != tt.wantNull {
				t.Errorf("IsNull() = %t, want %t", got, tt.wantNull)
			}
		})
	}
}
