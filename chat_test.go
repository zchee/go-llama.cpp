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
	"bytes"
	json "encoding/json/v2"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestMessageContentJSON(t *testing.T) {
	tests := map[string]struct {
		content  MessageContent
		json     string
		wantKind ContentKind
	}{
		"success: text":                    {content: TextContent("Hello"), json: `"Hello"`, wantKind: ContentKindText},
		"success: empty text is not unset": {content: TextContent(""), json: `""`, wantKind: ContentKindText},
		"success: parts": {
			content:  PartsContent(TextContentPart("What is this?"), ImageURLContentPart("data:image/png;base64,iVBORw==")),
			json:     `[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw=="}}]`,
			wantKind: ContentKindParts,
		},
		"success: media parts": {
			content: PartsContent(
				ContentPart{Type: "input_audio", InputAudio: &MediaInput{Data: new("UklGRg=="), Format: new("wav")}},
				ContentPart{Type: "video_url", VideoURL: &MediaInput{URL: new("https://example.com/v.mp4")}},
			),
			json:     `[{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}},{"type":"video_url","video_url":{"url":"https://example.com/v.mp4"}}]`,
			wantKind: ContentKindParts,
		},
		"success: empty parts": {content: PartsContent(), json: `[]`, wantKind: ContentKindParts},
		"success: null":        {content: NullContent(), json: `null`, wantKind: ContentKindNull},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := json.Marshal(tt.content, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.json {
				t.Errorf("Marshal = %s, want %s", got, tt.json)
			}
			var back MessageContent
			if err := json.Unmarshal(got, &back, jsonOptions); err != nil {
				t.Fatal(err)
			}
			if back.Kind() != tt.wantKind || back.IsZero() {
				t.Errorf("Kind = %d (zero %t), want %d", back.Kind(), back.IsZero(), tt.wantKind)
			}
			if diff := gocmp.Diff(tt.content, back); diff != "" {
				t.Errorf("round trip (-want +got):\n%s", diff)
			}
		})
	}
	for _, in := range []string{`3`, `true`, `{"type":"text"}`} {
		var c MessageContent
		if err := json.Unmarshal([]byte(in), &c); err == nil {
			t.Errorf("content accepted %s", in)
		}
	}
	if TextContent("x").Text() != "x" || len(PartsContent(TextContentPart("a")).Parts()) != 1 || !(MessageContent{}).IsZero() {
		t.Errorf("accessors do not return the built content")
	}
}

// A conversation the chat endpoints and /apply-template accept, as the openai package builds it.
const conversation = `[` +
	`{"role":"system","content":"be brief"},` +
	`{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"u","detail":"low"}}]},` +
	`{"role":"assistant","content":null,"reasoning_content":"r","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},` +
	`{"role":"tool","tool_call_id":"call_1","content":"42"},` +
	`{"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":""}}]}` +
	`]`

func TestMessageWire(t *testing.T) {
	data := []byte(conversation)
	got := jsontest.CheckWire[[]Message](t, data)
	jsontest.CheckFallbacks[Message](t)
	jsontest.CheckFallbacks[ContentPart](t)
	if len(*got) != 5 || (*got)[2].Content.Kind() != ContentKindNull || !(*got)[4].Content.IsZero() {
		t.Errorf("decoded conversation: %+v", *got)
	}
	if args := (*got)[4].ToolCalls[0].Function.Arguments; args == nil || *args != "" {
		t.Errorf("an empty streamed arguments delta must survive as an empty string, got %v", args)
	}
}

// TestNoUnknownSeesInsideMessageContent checks that the unknown-member gate reaches the parts of a
// MessageContent, which are kept in an unexported field.
func TestNoUnknownSeesInsideMessageContent(t *testing.T) {
	tests := map[string]struct {
		data string
		want string
	}{
		"error: content part":      {data: `[{"type":"text","text":"x"},{"type":"text","text":"y","future":1}]`, want: "/1: future"},
		"error: image of a part":   {data: `[{"type":"image_url","image_url":{"url":"u","x":1}}]`, want: "/0/image_url: x"},
		"success: nothing unknown": {data: `[{"type":"text","text":"x"}]`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			msgs, _ := jsontest.Failures(t, func(tb testing.TB) { jsontest.NoUnknown[MessageContent](tb, []byte(tt.data), nil) })
			if tt.want == "" {
				if len(msgs) != 0 {
					t.Errorf("unexpected failures: %v", msgs)
				}
				return
			}
			if len(msgs) != 1 || !strings.Contains(msgs[0], tt.want) {
				t.Errorf("failures = %v, want one naming %q", msgs, tt.want)
			}
		})
	}
}

func FuzzMessageContent(f *testing.F) {
	for _, seed := range []string{
		`"text"`, `null`, `[]`, `[{"type":"text","text":"x"}]`,
		`[{"type":"image_url","image_url":{"url":"u","detail":"auto"},"extra":[1]}]`,
		`[{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}]`,
		`[{}]`, `[{"type":""}]`, `[{"image_url":{}}]`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, in string) {
		var c MessageContent
		if err := json.Unmarshal([]byte(in), &c, jsonOptions); err != nil {
			return
		}
		out, err := json.Marshal(c, jsonOptions)
		if err != nil {
			t.Fatalf("decoded %s but cannot encode it: %v", in, err)
		}
		var back MessageContent
		if err := json.Unmarshal(out, &back, jsonOptions); err != nil {
			t.Fatalf("re-decoding %s: %v", out, err)
		}
		if !c.Equal(back) {
			t.Fatalf("round trip changed the content: %s -> %s", in, out)
		}
		// The encoding is stable; an explicit empty type, role or url is the one spelling it does not keep (see
		// TestEmptyChatStringsAreOmitted).
		again, err := json.Marshal(back, jsonOptions)
		if err != nil || !bytes.Equal(again, out) {
			t.Fatalf("re-encoding changed the content: %s -> %s -> %s", in, out, again)
		}
	})
}

// TestEmptyChatStringsAreOmitted pins the one normalisation of the chat types: an empty role, part type or image
// url is encoded as absent. The server reads an absent member and an empty string alike (json_value with a ""
// default in oaicompat_chat_params_parse), and streamed chat deltas omit role after the first chunk, which a
// re-encode must not turn into "role":"".
func TestEmptyChatStringsAreOmitted(t *testing.T) {
	tests := map[string]struct {
		in, want string
	}{
		"success: absent stays absent":   {in: `[{"role":"assistant"},{"content":"x"}]`, want: `[{"role":"assistant"},{"content":"x"}]`},
		"normalised: empty role":         {in: `[{"role":"","content":"x"}]`, want: `[{"content":"x"}]`},
		"normalised: empty type and url": {in: `[{"content":[{"type":"","image_url":{"url":""}}]}]`, want: `[{"content":[{"image_url":{}}]}]`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var msgs []Message
			if err := json.Unmarshal([]byte(tt.in), &msgs, jsonOptions); err != nil {
				t.Fatal(err)
			}
			out, err := json.Marshal(msgs, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			if string(out) != tt.want {
				t.Errorf("re-encoded %s as %s, want %s", tt.in, out, tt.want)
			}
		})
	}
}
