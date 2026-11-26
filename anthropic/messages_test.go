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
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"log"
	"net/http"
	"slices"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func checkAnthropicWire[T any](t *testing.T, data []byte) *T {
	t.Helper()
	got := jsontest.RoundTrip[T](t, data)
	jsontest.InjectUnknown[T](t, data)
	jsontest.NoUnknown[T](t, data, nil)
	jsontest.CheckFallbacks[T](t)
	return got
}

func TestAnthropicMessagesWire(t *testing.T) {
	tests := map[string]struct {
		prefix            string
		blocks            []string
		deltas            []string
		count, imageError bool
	}{
		"success: plain reply":                  {prefix: "messages/plain", blocks: []string{"text"}},
		"success: matched stop":                 {prefix: "messages/stop", blocks: []string{"text"}},
		"success: reasoning and reply":          {prefix: "messages/thinking", blocks: []string{"thinking", "text"}},
		"success: tool call":                    {prefix: "messages/tool", blocks: []string{"tool_use"}},
		"success: text stream":                  {prefix: "messages/stream", blocks: []string{"text"}, deltas: []string{"text_delta"}},
		"success: reasoning stream":             {prefix: "messages/thinking_stream", blocks: []string{"thinking", "text"}, deltas: []string{"thinking_delta", "text_delta", "signature_delta"}},
		"success: tool argument stream":         {prefix: "messages/tool_stream", blocks: []string{"tool_use"}, deltas: []string{"input_json_delta"}},
		"success: count history with tools":     {prefix: "count_tokens/history", count: true},
		"success: count explicit nulls":         {prefix: "count_tokens/null", count: true},
		"success: count tool references":        {prefix: "count_tokens/references", count: true},
		"success: count raw tool result":        {prefix: "count_tokens/raw_result", count: true},
		"success: count empty tool result":      {prefix: "count_tokens/empty_result", count: true},
		"error: model cannot read base64 image": {prefix: "messages/image_error", imageError: true},
		"error: model cannot read URL image":    {prefix: "messages/url_error", imageError: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/" + tt.prefix
			checkAnthropicWire[MessagesRequest](t, jsontest.ReadFile(t, prefix+".req.json"))
			switch {
			case tt.count:
				c := checkAnthropicWire[CountTokensResponse](t, jsontest.ReadFile(t, prefix+".resp.json"))
				if c.InputTokens <= 0 {
					t.Errorf("input_tokens = %d, want a nonempty prompt", c.InputTokens)
				}
			case tt.imageError:
				raw := jsontest.ReadFile(t, prefix+".resp.json")
				apiErr := llama.ReadAPIError(&http.Response{StatusCode: 500, Body: io.NopCloser(bytes.NewReader(raw))})
				if apiErr.Code != 500 || apiErr.Type != llama.ErrorTypeServer || !strings.Contains(apiErr.Message, "image input is not supported") {
					t.Fatalf("unexpected image error: %v", apiErr)
				}
			case tt.deltas != nil:
				checkAnthropicStream(t, jsontest.ReadFile(t, prefix+".resp.sse"), tt.blocks, tt.deltas)
			default:
				r := checkAnthropicWire[MessageResponse](t, jsontest.ReadFile(t, prefix+".resp.json"))
				if r.Type != "message" || r.Role != "assistant" || r.ID == "" || r.Model != "stories" || r.Usage.OutputTokens <= 0 {
					t.Errorf("unexpected message: %+v", r)
				}
				var blocks []string
				for i := range r.Content {
					blocks = append(blocks, r.Content[i].Type)
				}
				if diff := gocmp.Diff(tt.blocks, blocks); diff != "" {
					t.Errorf("content block types (-want +got):\n%s", diff)
				}
			}
		})
	}
	jsontest.CheckFallbacks[ContentBlock](t)
}

func checkAnthropicStream(t *testing.T, data []byte, wantBlocks, wantDeltas []string) {
	t.Helper()
	if bytes.Contains(data, []byte("data: [DONE]")) {
		t.Fatal("Anthropic stream has a [DONE] frame")
	}
	frames := jsontest.EventData(t, data)
	var eventTypes, blocks, deltas []string
	for ev, err := range llama.DecodeStream[MessageEvent](bytes.NewReader(data), 0) {
		if err != nil {
			t.Fatal(err)
		}
		i := len(eventTypes)
		if i >= len(frames) || frames[i].Type != ev.Type {
			t.Fatalf("event %d: SSE name does not match payload type %q", i, ev.Type)
		}
		want := checkAnthropicWire[MessageEvent](t, frames[i].Data)
		if diff := gocmp.Diff(want, ev, gocmp.AllowUnexported(Content{}, ContentBlock{})); diff != "" {
			t.Errorf("stream versus JSON decoder (-want +got):\n%s", diff)
		}
		eventTypes = append(eventTypes, ev.Type)
		switch ev.Type {
		case "content_block_start":
			blocks = append(blocks, ev.ContentBlock.Type)
		case "content_block_delta":
			if !slices.Contains(deltas, *ev.Delta.Type) {
				deltas = append(deltas, *ev.Delta.Type)
			}
		case "message_delta":
			if ev.Delta.StopDetails == nil || ev.Delta.StopReason == "" || ev.Usage.OutputTokens <= 0 {
				t.Fatalf("missing stop details or output tokens: %+v", ev)
			}
		}
	}
	if len(eventTypes) < 2 || len(frames) != len(eventTypes) || eventTypes[0] != "message_start" || eventTypes[len(eventTypes)-1] != "message_stop" {
		t.Fatalf("incomplete stream: %v", eventTypes)
	}
	if diff := gocmp.Diff(wantBlocks, blocks); diff != "" {
		t.Errorf("blocks (-want +got):\n%s", diff)
	}
	slices.Sort(deltas)
	if diff := gocmp.Diff(slices.Sorted(slices.Values(wantDeltas)), deltas); diff != "" {
		t.Errorf("deltas (-want +got):\n%s", diff)
	}
}

func TestAnthropicNewRequest(t *testing.T) {
	c, err := llama.New("http://localhost/prefix", llama.WithAPIKey("example"), llama.WithAPIKeyHeader(llama.APIKeyHeaderXAPIKey))
	if err != nil {
		t.Fatal(err)
	}
	body := &MessagesRequest{Model: new("stories"), Messages: []Message{{Role: "user", Content: TextContent("")}}, System: BlocksContent(), ExtraParams: &ExtraParams{Seed: new(int64(0)), CachePrompt: new(false)}}
	tests := map[string]struct {
		path   string
		stream *bool
	}{
		"success: request reply":  {path: "/v1/messages", stream: new(false)},
		"success: request stream": {path: "/v1/messages", stream: new(true)},
		"success: count tokens":   {path: "/v1/messages/count_tokens"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			r := body
			if tt.stream != nil {
				r = body.withStream(*tt.stream)
			}
			req, err := c.NewRequest(t.Context(), http.MethodPost, tt.path, r, llama.WithAutoload(false))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != "POST" || req.URL.RequestURI() != "/prefix"+tt.path+"?autoload=false" || req.Header.Get("X-Api-Key") == "" || req.Header.Get("Authorization") != "" {
				t.Errorf("wrong request method, target or auth header: %s %s", req.Method, req.URL.RequestURI())
			}
			raw, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			var got map[string]jsontext.Value
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatal(err)
			}
			want := map[string]jsontext.Value{"model": jsontext.Value(`"stories"`), "messages": jsontext.Value(`[{"role":"user","content":""}]`), "system": jsontext.Value(`[]`), "extra_params": jsontext.Value(`{"cache_prompt":false,"seed":0}`)}
			if tt.stream != nil {
				want["stream"] = jsontext.Value(fmt.Sprint(*tt.stream))
			}
			if diff := gocmp.Diff(want, got); diff != "" {
				t.Errorf("request body (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAnthropicContent(t *testing.T) {
	tests := map[string]struct {
		input   string
		want    *Content
		invalid bool
	}{
		"success: empty text":          {input: `""`, want: new(TextContent(""))},
		"success: empty array":         {input: `[]`, want: new(BlocksContent())},
		"success: null":                {input: `null`, want: new(NullContent())},
		"success: nested text":         {input: `[{"type":"tool_result","content":"ok"}]`, want: new(BlocksContent(ContentBlock{Type: "tool_result", Content: TextContent("ok")}))},
		"success: untagged block":      {input: `[{"text":"ignored"}]`, want: new(BlocksContent(ContentBlock{Text: new("ignored")}))},
		"success: null block":          {input: `[null,{"type":"text","text":"hello"}]`, want: new(BlocksContent(ContentBlock{null: true}, ContentBlock{Type: "text", Text: new("hello")}))},
		"success: raw object result":   {input: `[{"type":"tool_result","content":{"a":1}}]`},
		"success: raw number result":   {input: `[{"type":"tool_result","content":17}]`},
		"success: raw boolean result":  {input: `[{"type":"tool_result","content":false}]`},
		"success: raw array result":    {input: `[{"type":"tool_result","content":[2,false,[],null]}]`},
		"error: malformed raw content": {input: `{"a":`, invalid: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var got Content
			err := json.Unmarshal([]byte(tt.input), &got)
			if tt.invalid {
				if err == nil {
					t.Fatal("accepted invalid content")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tt.want != nil {
				if diff := gocmp.Diff(*tt.want, got, gocmp.AllowUnexported(Content{}, ContentBlock{})); diff != "" {
					t.Errorf("content (-want +got):\n%s", diff)
				}
			}
			jsontest.RoundTrip[Content](t, []byte(tt.input))
		})
	}
}

func FuzzAnthropicContent(f *testing.F) {
	for _, name := range []string{"messages/plain", "messages/image_error", "count_tokens/history", "count_tokens/null", "count_tokens/references", "count_tokens/raw_result"} {
		var seed struct {
			System   jsontext.Value `json:"system"`
			Messages []struct {
				Content jsontext.Value `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/"+name+".req.json"), &seed); err != nil {
			f.Fatal(err)
		}
		if len(seed.System) > 0 {
			f.Add([]byte(seed.System))
		}
		for _, m := range seed.Messages {
			f.Add([]byte(m.Content))
		}
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var got, back Content
		if err := json.Unmarshal(data, &got); err != nil {
			return
		}
		encoded, err := json.Marshal(got, json.Deterministic(true))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(encoded, &back); err != nil {
			t.Fatal(err)
		}
		// Compare the decoded values, ignoring insignificant whitespace or escaping in raw JSON members.
		normalize := gocmp.Transformer("RawJSON", func(v jsontext.Value) string {
			b, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			return string(b)
		})
		if diff := gocmp.Diff(got, back, gocmp.AllowUnexported(Content{}, ContentBlock{}), normalize); diff != "" {
			t.Fatalf("content changed on round trip (-decoded +redecoded):\n%s", diff)
		}
	})
}

func ExampleClient_Messages() {
	c, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		log.Fatal(err)
	}
	res, err := NewClient(c).Messages(context.Background(), &MessagesRequest{Messages: []Message{{Role: "user", Content: TextContent("Tell a short story.")}}, MaxTokens: new(64)})
	if err != nil {
		log.Fatal(err)
	}
	for i := range res.Content {
		if text := res.Content[i].Text; text != nil {
			fmt.Print(*text)
		}
	}
}
