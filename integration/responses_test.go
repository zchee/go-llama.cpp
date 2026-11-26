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

package integration

import (
	"encoding/json/jsontext"
	"errors"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

func TestResponsesStream(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	oc := openai.NewClient(c)
	req := &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("Once upon a time")}, MaxOutputTokens: new(4), Temperature: new(0.0), Stream: new(false), ExtraParams: &openai.ResponseExtraParams{Extra: map[string]jsontext.Value{"stream": jsontext.Value("false")}}}
	var first, last string
	var delta strings.Builder
	var completed *openai.Response
	for e, err := range oc.ResponsesStream(t.Context(), req, llama.WithAutoload(false)) {
		if err != nil {
			t.Fatal(err)
		}
		last = e.ResponseType()
		if first == "" {
			first = last
		}
		switch v := e.Value.(type) {
		case *openai.ResponseDeltaEvent:
			if v.Type == "response.output_text.delta" {
				delta.WriteString(v.Delta)
			}
		case *openai.ResponseLifecycleEvent:
			if v.Type == "response.completed" {
				completed = &v.Response
			}
		}
	}
	if first != "response.created" || last != "response.completed" || completed == nil || len(completed.Output) != 1 {
		t.Fatalf("unexpected streamed response: first %q, last %q, final %+v", first, last, completed)
	}
	msg, ok := completed.Output[0].Value.(*openai.ResponseMessage)
	if !ok || len(msg.Content.List) != 1 {
		t.Fatalf("unexpected completed item: %+v", completed.Output[0])
	}
	part, ok := msg.Content.List[0].Value.(*openai.ResponseTextPart)
	if !ok {
		t.Fatalf("unexpected completed part: %+v", msg.Content.List[0])
	}
	if got := delta.String(); got != part.Text {
		t.Errorf("streamed text = %q, completed text = %q", got, part.Text)
	}
	if *req.Stream || string(req.ExtraParams.Extra["stream"]) != "false" {
		t.Fatal("streaming modified the caller's request")
	}
	count, err := oc.ResponsesInputTokens(t.Context(), req)
	if err != nil {
		t.Fatal(err)
	}
	if completed.Usage == nil || completed.Usage.InputTokens != count.InputTokens {
		t.Fatalf("stream usage %+v disagrees with count %+v", completed.Usage, count)
	}
}

func TestResponsesAliases(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	req := &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("Once upon a time")}, MaxOutputTokens: new(4)}
	count, err := llama.Call[openai.InputTokensResponse](t.Context(), c, http.MethodPost, "/responses/input_tokens", req)
	if err != nil {
		t.Fatal(err)
	}
	response, err := llama.Call[openai.Response](t.Context(), c, http.MethodPost, "/responses", req)
	if err != nil {
		t.Fatal(err)
	}
	if count.InputTokens <= 0 || response.Status != "completed" || response.Usage == nil || response.Usage.InputTokens != count.InputTokens {
		t.Fatalf("alias completion usage %+v, alias count %+v", response.Usage, count)
	}
}

func TestResponsesRejectPreviousID(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	oc := openai.NewClient(c)
	req := &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("hello")}, Extra: map[string]jsontext.Value{"previous_response_id": jsontext.Value(`"resp_missing"`)}}
	tests := map[string]struct{ call func() error }{
		"error: completion cannot resolve previous response": {func() error { _, err := oc.Responses(t.Context(), req); return err }},
		"error: count cannot resolve previous response":      {func() error { _, err := oc.ResponsesInputTokens(t.Context(), req); return err }},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			e, ok := errors.AsType[*llama.APIError](tt.call())
			if !ok || e.StatusCode != http.StatusBadRequest || !strings.Contains(e.Message, "previous_response_id") {
				t.Fatalf("expected unsupported previous response error, got %v", e)
			}
		})
	}
}
