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
	"context"
	"encoding/json/jsontext"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/anthropic"
	"github.com/zchee/go-llama.cpp/openai"
)

func TestRequestGuard(t *testing.T) {
	c, obs := presetStories260K.server(t).Client(t)

	t.Run("error: openai completions with stream in Extra", func(t *testing.T) {
		_, err := openai.NewClient(c).Completions(t.Context(), &openai.CompletionRequest{
			Prompt: llama.TextPrompt("Once"), MaxTokens: new(1),
			Extra: map[string]jsontext.Value{"stream": jsontext.Value(`true`)},
		})
		if err == nil || !strings.Contains(err.Error(), "Stream") {
			t.Errorf("Completions error = %v, want an error naming Stream", err)
		}
	})

	t.Run("error: llama.Call with a promoted member in Extra", func(t *testing.T) {
		_, err := llama.Call[jsontext.Value](t.Context(), c, http.MethodPost, "/completion", &llama.CompletionRequest{
			Prompt: llama.TextPrompt("Once"),
			Extra:  map[string]jsontext.Value{"n_predict": jsontext.Value(`1`)},
		})
		if err == nil || !strings.Contains(err.Error(), "NPredict") {
			t.Errorf("Call error = %v, want an error naming NPredict", err)
		}
	})

	if got := len(obs.Exchanges()); got != 0 {
		t.Errorf("refused requests reached the server: %d exchanges, want none", got)
	}
}

func TestEndpointExtra(t *testing.T) {
	srv := presetStories260K.server(t)
	type extra = map[string]jsontext.Value
	stream := extra{"stream": jsontext.Value(`true`)}
	format := extra{"response_format": jsontext.Value(`"pcm"`)}
	chat := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := openai.NewClient(c).ChatCompletions(ctx, &openai.ChatCompletionRequest{Extra: e})
		return r == nil, err
	}
	responses := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := openai.NewClient(c).Responses(ctx, &openai.ResponsesRequest{Input: openai.ResponseInput{Text: new("Once")}, Extra: e})
		return r == nil, err
	}
	messages := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := anthropic.NewClient(c).Messages(ctx, &anthropic.MessagesRequest{MaxTokens: new(1), Extra: e})
		return r == nil, err
	}
	tts := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := c.TTS(ctx, &llama.TTSRequest{Input: "Once", Extra: e})
		if r != nil {
			r.Body.Close()
		}
		return r == nil, err
	}
	speech := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := openai.NewClient(c).Speech(ctx, &openai.SpeechRequest{Input: "Once", Extra: e})
		if r != nil {
			r.Body.Close()
		}
		return r == nil, err
	}
	rerank := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := c.Rerank(ctx, &llama.RerankRequest{Query: llama.RerankText("Once"), Documents: []llama.RerankInput{llama.RerankText("Once")}, Extra: e})
		return r == nil, err
	}
	rerankTEI := func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
		r, err := c.RerankTEI(ctx, &llama.RerankTEIRequest{Query: llama.RerankText("Once"), Texts: []llama.RerankInput{llama.RerankText("Once")}, Extra: e})
		return r == nil, err
	}
	topN := extra{"top_n": jsontext.Value(`4294967296`)}
	tests := map[string]struct {
		call  func(context.Context, *llama.Client, extra) (bool, error)
		extra extra
		field string
	}{
		"error: chat stream":               {chat, stream, "Stream"},
		"error: responses stream":          {responses, stream, "Stream"},
		"error: messages stream":           {messages, stream, "Stream"},
		"error: messages extra parameters": {messages, extra{"extra_params": jsontext.Value(`{"stream":true}`)}, "ExtraParams"},
		"error: native speech format":      {tts, format, "ResponseFormat"},
		"error: native speech stream":      {tts, stream, "Stream"},
		"error: openai speech format":      {speech, format, "ResponseFormat"},
		"error: openai speech stream":      {speech, stream, "Stream"},
		"error: rerank top n":              {rerank, topN, "TopN"},
		"error: TEI rerank top n":          {rerankTEI, topN, "TopN"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, obs := srv.Client(t)
			nilResponse, err := tt.call(t.Context(), c, tt.extra)
			if !nilResponse || err == nil || !strings.Contains(err.Error(), "use field "+tt.field) {
				t.Errorf("nil response = %t, error = %v; want nil and error naming %s", nilResponse, err, tt.field)
			}
			if got := len(obs.Exchanges()); got != 0 {
				t.Errorf("sent %d requests; want none", got)
			}
		})
	}
}

func TestCallDiscardsPartialResponse(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	type invalidHealth struct {
		Status int `json:"status"`
	}
	got, err := llama.Call[*invalidHealth](t.Context(), c, http.MethodGet, "/health", nil)
	if err == nil || got != nil {
		t.Errorf("Call = (%+v, %v), want nil and a decode error", got, err)
	}
}

func TestGrammarTakesPrecedenceOverJSONSchema(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	const grammar = `root ::= "hello"`
	res, err := c.Completion(t.Context(), &llama.CompletionRequest{
		Prompt: llama.TextPrompt("Once"), NPredict: new(1), Grammar: new(grammar),
		JSONSchema: jsontext.Value(`{"type":"boolean"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.GenerationSettings == nil || res.GenerationSettings.Grammar == nil || *res.GenerationSettings.Grammar != grammar {
		t.Fatalf("effective settings = %+v, want explicit grammar %q", res.GenerationSettings, grammar)
	}
}
