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

package openai_test

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"log"
	"net/http"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/openai"
)

func responsesWire[T any](t testing.TB, data []byte) *T {
	t.Helper()
	v := jsontest.RoundTrip[T](t, data)
	jsontest.InjectUnknown[T](t, data)
	jsontest.NoUnknown[T](t, data, nil)
	jsontest.CheckFallbacks[T](t)
	return v
}

func TestResponsesWire(t *testing.T) {
	tests := map[string]struct {
		fixture         string
		isCount, stream bool
		errStatus       int
	}{
		"success: completed message":                      {fixture: "basic"},
		"success: completed function call":                {fixture: "tool"},
		"success: reasoning and message":                  {fixture: "reasoning"},
		"success: text events":                            {fixture: "stream", stream: true},
		"success: function argument events":               {fixture: "tool_stream", stream: true},
		"success: reasoning events":                       {fixture: "reasoning_stream", stream: true},
		"error: unsupported input file":                   {fixture: "error", errStatus: 400},
		"success: string input count":                     {fixture: "basic", isCount: true},
		"success: history with reasoning and tool output": {fixture: "history", isCount: true},
		"success: configured format and named tool":       {fixture: "options", isCount: true},
		"error: image without a projector":                {fixture: "image", isCount: true, errStatus: 500},
		"error: downstream parser rejects refusals":       {fixture: "refusal", isCount: true, errStatus: 400},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			group := "responses"
			if tt.isCount {
				group = "responses_input_tokens"
			}
			prefix := "testdata/" + group + "/" + tt.fixture
			responsesWire[openai.ResponsesRequest](t, jsontest.ReadFile(t, prefix+".req.json"))
			if tt.stream {
				responsesCheckStream(t, jsontest.ReadFile(t, prefix+".resp.sse"))
				return
			}
			data := jsontest.ReadFile(t, prefix+".resp.json")
			if tt.errStatus != 0 {
				e := llama.ReadAPIError(&http.Response{StatusCode: tt.errStatus, Body: io.NopCloser(bytes.NewReader(data))})
				if e.Code != tt.errStatus || e.Message == "" || e.Type == "" {
					t.Fatalf("unexpected API error: %+v", e)
				}
				return
			}
			if tt.isCount {
				responsesWire[openai.InputTokensResponse](t, data)
				return
			}
			r := responsesWire[openai.Response](t, data)
			if len(r.Output) == 0 {
				t.Fatal("empty output")
			}
			switch tt.fixture {
			case "tool":
				v, ok := r.Output[0].Value.(*openai.ResponseFunctionCall)
				if !ok || v.Name != "ping" || v.Arguments != "{}" {
					t.Fatalf("expected ping call, got %#v", r.Output[0].Value)
				}
			case "reasoning":
				v, ok := r.Output[0].Value.(*openai.ResponseReasoningItem)
				if !ok || len(v.Content) != 1 {
					t.Fatalf("expected reasoning content, got %#v", r.Output[0].Value)
				}
			}
		})
	}
}

func responsesCheckStream(t *testing.T, data []byte) {
	t.Helper()
	if bytes.Contains(data, []byte("[DONE]")) {
		t.Fatal("Responses stream unexpectedly has a [DONE] frame")
	}
	frames := jsontest.EventData(t, data)
	var names []string
	for e, err := range llama.DecodeStream[openai.ResponseEvent](bytes.NewReader(data), 0) {
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, e.ResponseType())
	}
	var want []string
	for _, frame := range frames {
		want = append(want, frame.Type)
		v := responsesWire[openai.ResponseEvent](t, frame.Data)
		if _, unknown := v.Value.(*openai.ResponseUnknown); unknown {
			t.Errorf("untyped recorded event %q", frame.Type)
		}
	}
	if diff := gocmp.Diff(want, names); diff != "" {
		t.Fatalf("event names (-SSE +JSON):\n%s", diff)
	}
	if len(names) == 0 || names[0] != "response.created" || names[len(names)-1] != "response.completed" {
		t.Fatalf("unexpected event ordering: %v", names)
	}
}

func TestResponsesUnknownKinds(t *testing.T) {
	// Give future kinds incompatible values for fields used by today's kinds: decoding them as
	// known variants would either fail or change the payload.
	data := []byte(`{"content":false,"delta":{"nested":1},"type":"future","item_id":[2],"arguments":17}`)
	tests := map[string]struct {
		unknown func() *openai.ResponseUnknown
	}{
		"success: future item": {func() *openai.ResponseUnknown {
			v, _ := jsontest.RoundTrip[openai.ResponseItem](t, data).Value.(*openai.ResponseUnknown)
			return v
		}},
		"success: future event": {func() *openai.ResponseUnknown {
			v, _ := jsontest.RoundTrip[openai.ResponseEvent](t, data).Value.(*openai.ResponseUnknown)
			return v
		}},
		"success: future part": {func() *openai.ResponseUnknown {
			v, _ := jsontest.RoundTrip[openai.ResponsePart](t, data).Value.(*openai.ResponseUnknown)
			return v
		}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			v := tt.unknown()
			if v == nil || len(v.Extra) != 4 {
				t.Fatalf("expected opaque payload, got %+v", v)
			}
		})
	}
}

func TestResponsesNewRequest(t *testing.T) {
	c, err := llama.New("http://127.0.0.1:8080/prefix")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		path, fixture string
		body          openai.ResponsesRequest
	}{
		"success: create with explicit zero temperature": {path: "/v1/responses", fixture: "responses/basic", body: openai.ResponsesRequest{Model: new("responses-test"), Input: openai.ResponseInput{Text: new("Once upon a time")}, Instructions: new("Write briefly."), MaxOutputTokens: new(5), Temperature: new(0.0), ExtraParams: &openai.ResponseExtraParams{Seed: new(int64(42))}}},
		"success: count without generating":              {path: "/v1/responses/input_tokens", fixture: "responses_input_tokens/basic", body: openai.ResponsesRequest{Model: new("tinyllama-2"), Input: openai.ResponseInput{Text: new("Hello there")}}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, tt.path, &tt.body, llama.WithAutoload(false))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != http.MethodPost || req.URL.Path != "/prefix"+tt.path || req.URL.Query().Get("autoload") != "false" {
				t.Fatalf("unexpected request: %s %s", req.Method, req.URL)
			}
			got, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			want := jsontext.Value(jsontest.ReadFile(t, "testdata/"+tt.fixture+".req.json"))
			actual := jsontext.Value(got)
			if err := want.Canonicalize(); err != nil {
				t.Fatal(err)
			}
			if err := actual.Canonicalize(); err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(string(want), string(actual)); diff != "" {
				t.Errorf("request (-recorded +created):\n%s", diff)
			}
		})
	}
}

func ExampleClient_Responses() {
	c, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		log.Fatal(err)
	}
	r, err := openai.NewClient(c).Responses(context.Background(), &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("How does a sundial work?")}})
	if err != nil {
		log.Fatal(err)
	}
	for _, item := range r.Output {
		if msg, ok := item.Value.(*openai.ResponseMessage); ok {
			for _, part := range msg.Content.List {
				if text, ok := part.Value.(*openai.ResponseTextPart); ok {
					fmt.Print(text.Text)
				}
			}
		}
	}
}

// Check the decoded value rather than the source bytes: a valid custom codec may normalize the spelling.
func responsesFuzz[T any](f *testing.F) {
	f.Helper()
	f.Fuzz(func(t *testing.T, data []byte) {
		var first, second T
		if err := json.Unmarshal(data, &first); err != nil {
			return
		}
		out, err := json.Marshal(first, json.Deterministic(true))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(out, &second); err != nil {
			t.Fatal(err)
		}
		compactRaw := gocmp.Transformer("CompactJSON", func(v jsontext.Value) string {
			b, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			return string(b)
		})
		if diff := gocmp.Diff(first, second, compactRaw); diff != "" {
			t.Fatalf("decoded values changed (-first +second):\n%s", diff)
		}
	})
}

func FuzzResponsesInput(f *testing.F) {
	for _, path := range []string{"responses/basic", "responses_input_tokens/history"} {
		var req struct{ Input jsontext.Value }
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/"+path+".req.json"), &req, json.MatchCaseInsensitiveNames(true)); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(req.Input))
	}
	responsesFuzz[openai.ResponseInput](f)
}

func FuzzResponsesItem(f *testing.F) {
	for _, file := range []string{"tool", "reasoning"} {
		var r struct {
			Output []jsontext.Value `json:"output"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/responses/"+file+".resp.json"), &r); err != nil {
			f.Fatal(err)
		}
		for _, item := range r.Output {
			f.Add([]byte(item))
		}
	}
	responsesFuzz[openai.ResponseItem](f)
}

func FuzzResponsesContent(f *testing.F) {
	var r struct {
		Input []struct {
			Content jsontext.Value `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/responses_input_tokens/history.req.json"), &r); err != nil {
		f.Fatal(err)
	}
	for _, item := range r.Input {
		if len(item.Content) > 0 {
			f.Add([]byte(item.Content))
		}
	}
	responsesFuzz[openai.ResponseContent](f)
}

func FuzzResponsesPart(f *testing.F) {
	for _, file := range []string{"stream", "reasoning_stream"} {
		for _, frame := range jsontest.EventData(f, jsontest.ReadFile(f, "testdata/responses/"+file+".resp.sse")) {
			var e struct {
				Part jsontext.Value `json:"part"`
			}
			if err := json.Unmarshal(frame.Data, &e); err != nil {
				f.Fatal(err)
			}
			if len(e.Part) > 0 {
				f.Add([]byte(e.Part))
			}
		}
	}
	responsesFuzz[openai.ResponsePart](f)
}

func FuzzResponsesEvent(f *testing.F) {
	for _, file := range []string{"stream", "reasoning_stream", "tool_stream"} {
		for _, frame := range jsontest.EventData(f, jsontest.ReadFile(f, "testdata/responses/"+file+".resp.sse")) {
			f.Add(frame.Data)
		}
	}
	responsesFuzz[openai.ResponseEvent](f)
}
