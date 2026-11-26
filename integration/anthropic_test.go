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
	json "encoding/json/v2"
	"errors"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/anthropic"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

var presetAnthropic = &serverPreset{
	name: "stories-anthropic",
	options: func(t testing.TB) harness.Options {
		return harness.Options{Model: harness.Model(t, "stories260K"), Args: []string{"--alias", "stories", "--ctx-size", "512", "--parallel", "1", "--jinja", "--seed", "42"}, APIKey: testAPIKey}
	},
}

var anthropicClientOptions = []llama.ClientOption{llama.WithAPIKey(testAPIKey), llama.WithAPIKeyHeader(llama.APIKeyHeaderXAPIKey)}

func anthropicRequest() *anthropic.MessagesRequest {
	return &anthropic.MessagesRequest{Messages: []anthropic.Message{{Role: "user", Content: anthropic.TextContent("Tell a story.")}}, MaxTokens: new(8), Temperature: new(0.0)}
}

func TestAnthropicCountMatchesMessageUsage(t *testing.T) {
	c, obs := presetAnthropic.server(t).Client(t, anthropicClientOptions...)
	client := anthropic.NewClient(c)
	req := anthropicRequest()
	req.System = anthropic.BlocksContent(anthropic.ContentBlock{Type: "text", Text: new("Be brief.")})
	count, err := client.CountTokens(t.Context(), req)
	if err != nil {
		t.Fatal(err)
	}
	req.Stream = new(true)
	req.ExtraParams = &anthropic.ExtraParams{Extra: map[string]jsontext.Value{"stream": jsontext.Value(`true`)}}
	res, err := client.Messages(t.Context(), req, llama.WithAutoload(false))
	if err != nil {
		t.Fatal(err)
	}
	if got := int64(res.Usage.InputTokens + res.Usage.CacheReadInputTokens); got != count.InputTokens {
		t.Errorf("count = %d; message reports %d prompt tokens incl. cache", count.InputTokens, got)
	}
	if !*req.Stream || string(req.ExtraParams.Extra["stream"]) != "true" {
		t.Fatal("Messages changed stream options on its input")
	}
	ex := obs.Exchanges()
	if len(ex) != 2 || ex[1].Query.Get("autoload") != "false" || ex[1].Header.Get("X-Api-Key") == "" || ex[1].Header.Get("Authorization") != "" {
		t.Fatal("request options or X-Api-Key header did not reach the server")
	}
}

func TestAnthropicRawToolResult(t *testing.T) {
	c, _ := presetAnthropic.server(t).Client(t, anthropicClientOptions...)
	client := anthropic.NewClient(c)
	tests := map[string]struct{ fixture string }{
		"success: object result with untagged and null blocks": {fixture: "raw_result"},
		"success: empty string result":                         {fixture: "empty_result"},
	}
	var tokenCount int64
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var req anthropic.MessagesRequest
			if err := json.Unmarshal(jsontest.ReadFile(t, "../anthropic/testdata/count_tokens/"+tt.fixture+".req.json"), &req); err != nil {
				t.Fatal(err)
			}
			count, err := client.CountTokens(t.Context(), &req)
			if err != nil {
				t.Fatal(err)
			}
			if count.InputTokens <= 0 {
				t.Fatalf("input_tokens = %d, want a counted prompt", count.InputTokens)
			}
			if tokenCount != 0 && tokenCount != count.InputTokens {
				t.Errorf("input_tokens = %d, want same total as the other result (%d)", count.InputTokens, tokenCount)
			}
			tokenCount = count.InputTokens
		})
	}
}

func TestAnthropicMessageStream(t *testing.T) {
	c, obs := presetAnthropic.server(t).Client(t, anthropicClientOptions...)
	req := anthropicRequest()
	req.ExtraParams = &anthropic.ExtraParams{Extra: map[string]jsontext.Value{"stream": jsontext.Value(`false`)}}
	seq := anthropic.NewClient(c).MessagesStream(t.Context(), req)
	if len(obs.Exchanges()) != 0 {
		t.Fatal("stream opened before iteration")
	}
	var types []string
	var text strings.Builder
	for ev, err := range seq {
		if err != nil {
			t.Fatal(err)
		}
		types = append(types, ev.Type)
		if ev.Delta != nil && ev.Delta.Text != nil {
			text.WriteString(*ev.Delta.Text)
		}
	}
	if len(types) < 2 || types[0] != "message_start" || types[len(types)-1] != "message_stop" || text.Len() == 0 {
		t.Fatalf("incomplete stream: events=%v text=%q", types, text.String())
	}
	if req.Stream != nil || string(req.ExtraParams.Extra["stream"]) != "false" {
		t.Fatal("MessagesStream changed its input")
	}
	var errs []error
	for _, err := range seq {
		errs = append(errs, err)
	}
	if len(errs) != 1 || !errors.Is(errs[0], llama.ErrStreamConsumed) {
		t.Fatalf("second iteration yielded %v", errs)
	}
	if ex := obs.Exchanges(); len(ex) != 1 || !ex[0].BodyClosed() {
		t.Fatal("stream did not close its only exchange")
	}
}

func TestAnthropicRequestErrors(t *testing.T) {
	tests := map[string]struct {
		opts   []llama.ClientOption
		text   string
		status int
		typ    llama.ErrorType
	}{
		"error: no API key":             {text: "Hello", status: http.StatusUnauthorized, typ: llama.ErrorTypeAuthentication},
		"error: prompt exceeds context": {opts: anthropicClientOptions, text: strings.Repeat("hello ", 1024), status: http.StatusBadRequest, typ: llama.ErrorTypeExceedContextSize},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, obs := presetAnthropic.server(t).Client(t, tt.opts...)
			req := anthropicRequest()
			req.Messages[0].Content = anthropic.TextContent(tt.text)
			var errs []error
			for ev, err := range anthropic.NewClient(c).MessagesStream(t.Context(), req) {
				if ev != nil {
					t.Fatalf("unexpected event before rejection: %+v", ev)
				}
				errs = append(errs, err)
			}
			if len(errs) != 1 {
				t.Fatalf("got %d errors, want one", len(errs))
			}
			e, ok := errors.AsType[*llama.APIError](errs[0])
			if !ok || e.StatusCode != tt.status || e.Type != tt.typ {
				t.Fatalf("stream error = %v, want HTTP %d %s", errs[0], tt.status, tt.typ)
			}
			if ex := obs.Exchanges(); len(ex) != 1 || !ex[0].BodyClosed() {
				t.Fatal("error response was not closed")
			}
		})
	}
}
