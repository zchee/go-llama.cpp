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

package openai

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestCompletionsResponseWire(t *testing.T) {
	tests := map[string]struct {
		path   string
		verify func(t *testing.T, got *CompletionResponse)
	}{
		"success: basic": {
			path: "testdata/completions/basic.resp.json",
			verify: func(t *testing.T, got *CompletionResponse) {
				if got.Object != "text_completion" {
					t.Errorf("Object = %q, want %q", got.Object, "text_completion")
				}
				if len(got.Choices) != 1 || got.Choices[0].Text == "" {
					t.Fatalf("Choices = %+v, want 1 non-empty choice", got.Choices)
				}
				if got.Choices[0].FinishReason == nil || *got.Choices[0].FinishReason != "length" {
					t.Errorf("FinishReason = %v, want %q", got.Choices[0].FinishReason, "length")
				}
				if got.Usage == nil || got.Usage.TotalTokens != got.Usage.PromptTokens+got.Usage.CompletionTokens {
					t.Errorf("Usage = %+v, want consistent token counts", got.Usage)
				}
				if got.Timings == nil || got.Timings.PredictedN == 0 {
					t.Errorf("Timings = %+v, want populated timings", got.Timings)
				}
			},
		},
		"success: logprobs": {
			path: "testdata/completions/logprobs.resp.json",
			verify: func(t *testing.T, got *CompletionResponse) {
				if len(got.Choices) != 1 {
					t.Fatalf("len(Choices) = %d, want 1", len(got.Choices))
				}
				lp := got.Choices[0].Logprobs
				if lp == nil || len(lp.Content) == 0 {
					t.Fatalf("Logprobs = %+v, want non-empty content", lp)
				}
				if lp.Content[0].Logprob == nil || len(lp.Content[0].TopLogprobs) == 0 {
					t.Errorf("Logprobs.Content[0] = %+v, want logprob and top_logprobs", lp.Content[0])
				}
			},
		},
		"success: multiple choices": {
			path: "testdata/completions/multi.resp.json",
			verify: func(t *testing.T, got *CompletionResponse) {
				if len(got.Choices) != 2 {
					t.Fatalf("len(Choices) = %d, want 2", len(got.Choices))
				}
				if got.Choices[0].Index != 0 || got.Choices[1].Index != 1 {
					t.Errorf("indices = (%d, %d), want (0, 1)", got.Choices[0].Index, got.Choices[1].Index)
				}
			},
		},
		"success: verbose": {
			path: "testdata/completions/verbose.resp.json",
			verify: func(t *testing.T, got *CompletionResponse) {
				if len(got.Verbose) == 0 || got.Verbose.Kind() != '{' {
					t.Errorf("Verbose = %s, want JSON object", got.Verbose)
				}
			},
		},
		"success: finish_reason stop": {
			path: "testdata/completions/stop.resp.json",
			verify: func(t *testing.T, got *CompletionResponse) {
				if len(got.Choices) != 1 {
					t.Fatalf("len(Choices) = %d, want 1", len(got.Choices))
				}
				if got.Choices[0].FinishReason == nil || *got.Choices[0].FinishReason != "stop" {
					t.Errorf("FinishReason = %v, want %q", got.Choices[0].FinishReason, "stop")
				}
			},
		},
		"success: max_completion_tokens and n_cmpl aliases": {
			path: "testdata/completions/aliases.resp.json",
			verify: func(t *testing.T, got *CompletionResponse) {
				// Recorded with max_completion_tokens=4 and n_cmpl=2: two choices of four tokens each.
				if len(got.Choices) != 2 {
					t.Fatalf("len(Choices) = %d, want 2", len(got.Choices))
				}
				if got.Usage == nil || got.Usage.CompletionTokens != 8 {
					t.Errorf("Usage = %+v, want 8 completion tokens summed over both choices", got.Usage)
				}
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, tt.path)
			got := jsontest.RoundTrip[CompletionResponse](t, data)
			jsontest.InjectUnknown[CompletionResponse](t, data)
			jsontest.NoUnknown[CompletionResponse](t, data, nil)
			tt.verify(t, got)
		})
	}
}

func TestCompletionsStreamWire(t *testing.T) {
	tests := map[string]struct {
		path   string
		verify func(t *testing.T, chunks []*CompletionResponse)
	}{
		"success: basic stream": {
			path: "testdata/completions/stream.resp.sse",
			verify: func(t *testing.T, chunks []*CompletionResponse) {
				if len(chunks) < 2 {
					t.Fatalf("len(chunks) = %d, want >= 2", len(chunks))
				}
				var text strings.Builder
				for _, ch := range chunks {
					if ch.Object != "text_completion" {
						t.Errorf("chunk Object = %q, want %q", ch.Object, "text_completion")
					}
					for _, choice := range ch.Choices {
						text.WriteString(choice.Text)
					}
				}
				if text.Len() == 0 {
					t.Errorf("accumulated stream text is empty")
				}
				last := chunks[len(chunks)-1]
				if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "length" {
					t.Errorf("last FinishReason = %v, want %q", last.Choices[0].FinishReason, "length")
				}
				if last.Usage == nil || last.Timings == nil {
					t.Errorf("last chunk missing Usage (%v) or Timings (%v)", last.Usage, last.Timings)
				}
			},
		},
		"success: include_usage false does not suppress usage": {
			path: "testdata/completions/stream_no_usage.resp.sse",
			verify: func(t *testing.T, chunks []*CompletionResponse) {
				if len(chunks) < 2 {
					t.Fatalf("len(chunks) = %d, want >= 2", len(chunks))
				}
				// Recorded with stream_options.include_usage=false: the final chunk still carries usage,
				// because the server writes it unconditionally on this route.
				last := chunks[len(chunks)-1]
				if last.Usage == nil {
					t.Errorf("last chunk = %+v, want usage despite include_usage=false", last)
				}
				if len(last.Choices) != 1 || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "length" {
					t.Errorf("last chunk choices = %+v, want finish_reason=length", last.Choices)
				}
			},
		},
		"success: partial __verbose chunks": {
			path: "testdata/completions/stream_verbose.resp.sse",
			verify: func(t *testing.T, chunks []*CompletionResponse) {
				if len(chunks) < 2 {
					t.Fatalf("len(chunks) = %d, want >= 2", len(chunks))
				}
				// With verbose set, every chunk carries __verbose: intermediate chunks hold the partial native
				// response and the final chunk the full one, with generation_settings.
				for i, ch := range chunks {
					if len(ch.Verbose) == 0 || ch.Verbose.Kind() != '{' {
						t.Errorf("chunk %d Verbose = %s, want JSON object", i, ch.Verbose)
					}
				}
				partial := chunks[0]
				if bytes.Contains(partial.Verbose, []byte(`"generation_settings"`)) {
					t.Errorf("first chunk Verbose = %s, want a partial object without generation_settings", partial.Verbose)
				}
				final := chunks[len(chunks)-1]
				if !bytes.Contains(final.Verbose, []byte(`"generation_settings"`)) {
					t.Errorf("final chunk Verbose = %s, want the full object with generation_settings", final.Verbose)
				}
			},
		},
		"success: progress, timings_per_token, and post_sampling_probs": {
			path: "testdata/completions/stream_full.resp.sse",
			verify: func(t *testing.T, chunks []*CompletionResponse) {
				var sawProgress, sawProb, sawTimings bool
				for _, ch := range chunks {
					if ch.PromptProgress != nil {
						sawProgress = true
					}
					if ch.Timings != nil {
						sawTimings = true
					}
					if len(ch.Choices) > 0 && ch.Choices[0].Logprobs != nil && len(ch.Choices[0].Logprobs.Content) > 0 {
						tp := ch.Choices[0].Logprobs.Content[0]
						if tp.Prob != nil && len(tp.TopProbs) > 0 {
							sawProb = true
						}
					}
				}
				if !sawProgress || !sawProb || !sawTimings {
					t.Errorf("sawProgress=%t sawProb=%t sawTimings=%t, want all true", sawProgress, sawProb, sawTimings)
				}
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			raw := jsontest.ReadFile(t, tt.path)
			for i, ev := range jsontest.EventData(t, raw) {
				t.Run(fmt.Sprintf("event_%d", i), func(t *testing.T) {
					jsontest.RoundTrip[CompletionResponse](t, ev.Data)
					jsontest.InjectUnknown[CompletionResponse](t, ev.Data)
					jsontest.NoUnknown[CompletionResponse](t, ev.Data, nil)
				})
			}

			var chunks []*CompletionResponse
			for ch, err := range llama.DecodeStream[CompletionResponse](bytes.NewReader(raw), 0) {
				if err != nil {
					t.Fatalf("DecodeStream error: %v", err)
				}
				chunks = append(chunks, ch)
			}
			tt.verify(t, chunks)
		})
	}
}

func TestCompletionsRequestWire(t *testing.T) {
	fixtures := map[string]string{
		"success: basic":           "testdata/completions/basic.req.json",
		"success: stream":          "testdata/completions/stream.req.json",
		"success: logprobs":        "testdata/completions/logprobs.req.json",
		"success: multi":           "testdata/completions/multi.req.json",
		"success: stream_full":     "testdata/completions/stream_full.req.json",
		"success: verbose":         "testdata/completions/verbose.req.json",
		"success: error":           "testdata/completions/error.req.json",
		"success: aliases":         "testdata/completions/aliases.req.json",
		"success: stop":            "testdata/completions/stop.req.json",
		"success: stream_verbose":  "testdata/completions/stream_verbose.req.json",
		"success: stream_no_usage": "testdata/completions/stream_no_usage.req.json",
		"success: stream_error":    "testdata/completions/stream_error.req.json",
	}
	for name, path := range fixtures {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, path)
			jsontest.RoundTrip[CompletionRequest](t, data)
			jsontest.InjectUnknown[CompletionRequest](t, data)
			jsontest.NoUnknown[CompletionRequest](t, data, nil)
		})
	}

	variants := map[string]struct {
		req  CompletionRequest
		skip []string
	}{
		"success: token prompt and sampling extensions": {
			req: CompletionRequest{
				Model:               "tinyllama-2",
				Prompt:              llama.TokenPrompt(1, 2, 3),
				MaxTokens:           new(16),
				MaxCompletionTokens: new(8),
				N:                   new(2),
				NCmpl:               new(2),
				Logprobs:            new(3),
				Stream:              new(true),
				StreamOptions:       &StreamOptions{IncludeUsage: new(true)},
				Temperature:         new(0.7),
				Mirostat:            new(2),
				Seed:                new(int64(42)),
				Stop:                llama.StringSlice("\n", "END"),
				Samplers:            llama.StringValue("kpt"),
				LogitBias:           llama.LogitBiasObject(llama.LogitBiasEntry{Target: llama.TokenTarget(1), Bias: -5}, llama.LogitBiasEntry{Target: llama.TextTarget("bad"), Ban: true}),
				JSONSchema:          jsontext.Value(`{"type":"object"}`),
				LoRA:                []llama.LoRAWeight{{ID: 0, Scale: 0.5}},
			},
			skip: []string{
				// logit_bias in object form is a JSON map from token/string to bias, not a record struct.
				"/logit_bias",
			},
		},
		"success: mixed prompt": {
			req: CompletionRequest{
				Prompt: llama.MixedPrompt(llama.TokenPart(1), llama.TextPart("hello")),
			},
		},
		"success: batch prompt": {
			req: CompletionRequest{
				Prompt: llama.BatchPrompt(llama.TextPrompt("a"), llama.TokenPrompt(1, 2)),
			},
		},
	}

	for name, tt := range variants {
		t.Run(name, func(t *testing.T) {
			data, err := json.Marshal(&tt.req, json.Deterministic(true))
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			got := jsontest.RoundTrip[CompletionRequest](t, data)
			if !tt.req.Prompt.Equal(got.Prompt) {
				t.Errorf("Prompt mismatch after round trip")
			}
			if diff := gocmp.Diff(tt.req.SamplingParams, got.SamplingParams,
				gocmp.Comparer(llama.Prompt.Equal),
				gocmp.Comparer(llama.LogitBias.Equal),
				gocmp.Comparer(llama.StringOrSlice.Equal),
			); diff != "" {
				t.Errorf("SamplingParams diff (-want +got):\n%s", diff)
			}
			jsontest.InjectUnknown[CompletionRequest](t, data, jsontest.SkipObjects(tt.skip...))
			jsontest.NoUnknown[CompletionRequest](t, data, nil)
		})
	}
}

// TestCompletionsMultimodalRefusalWire proves on recorded exchanges that an object prompt both round-trips through
// the Go type and is refused by this route: without a projector the server rejects multimodal data, and with one
// it reads the prompt as a plain string, so object and token-array prompts fail with "type must be string".
func TestCompletionsMultimodalRefusalWire(t *testing.T) {
	objectReq := CompletionRequest{
		Prompt: llama.MultimodalPrompt("<__media__> describe", []byte("png")),
	}
	data, err := json.Marshal(&objectReq, json.Deterministic(true))
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got := jsontest.RoundTrip[CompletionRequest](t, data)
	if !objectReq.Prompt.Equal(got.Prompt) {
		t.Errorf("Prompt mismatch after round trip")
	}

	tests := map[string]struct {
		prefix      string
		wantPrompt  llama.Prompt
		wantMessage string
	}{
		"error: multimodal data without a projector": {
			prefix:      "testdata/completions/error_multimodal",
			wantPrompt:  llama.MultimodalPrompt("<__media__> describe", []byte("png")),
			wantMessage: "does not support multimodal requests",
		},
		"error: object prompt with a projector": {
			prefix:      "testdata/completions/error_mmproj_object",
			wantPrompt:  llama.MultimodalPrompt("<__media__> describe", []byte("png")),
			wantMessage: "type must be string",
		},
		"error: token prompt with a projector": {
			prefix:      "testdata/completions/error_mmproj_tokens",
			wantPrompt:  llama.TokenPrompt(1, 2, 3),
			wantMessage: "type must be string",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			reqData := jsontest.ReadFile(t, tt.prefix+".req.json")
			var req CompletionRequest
			if err := json.Unmarshal(reqData, &req); err != nil {
				t.Fatalf("unmarshaling recorded request: %v", err)
			}
			if !req.Prompt.Equal(tt.wantPrompt) {
				t.Errorf("recorded prompt %v does not match the typed prompt", req.Prompt)
			}

			respData := jsontest.ReadFile(t, tt.prefix+".resp.json")
			apiErr := llama.ReadAPIError(&http.Response{
				StatusCode: http.StatusBadRequest,
				Header:     http.Header{"Content-Type": {"application/json; charset=utf-8"}},
				Body:       io.NopCloser(bytes.NewReader(respData)),
			})
			if apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest {
				t.Errorf("error = %v, want 400 %s", apiErr, llama.ErrorTypeInvalidRequest)
			}
			if !strings.Contains(apiErr.Message, tt.wantMessage) {
				t.Errorf("Message = %q, want it to contain %q", apiErr.Message, tt.wantMessage)
			}
		})
	}
}

// TestCompletionsStreamErrorWire replays a recorded stream in which an error frame arrives after HTTP 200: a batch
// of a short prompt and one over the context size, run on a server with one slot, streams the short prompt's
// result first and then the oversized prompt's error, which ends the stream without [DONE].
func TestCompletionsStreamErrorWire(t *testing.T) {
	raw := jsontest.ReadFile(t, "testdata/completions/stream_error.resp.sse")
	var chunks []*CompletionResponse
	var streamErr error
	for ch, err := range llama.DecodeStream[CompletionResponse](bytes.NewReader(raw), 0) {
		if err != nil {
			streamErr = err
			break
		}
		chunks = append(chunks, ch)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks decoded before the error frame")
	}
	last := chunks[len(chunks)-1]
	if last.Usage == nil || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "length" {
		t.Errorf("last chunk before the error = %+v, want a finished result with usage", last)
	}
	apiErr, ok := errors.AsType[*llama.APIError](streamErr)
	if !ok {
		t.Fatalf("stream error = %v, want *llama.APIError", streamErr)
	}
	if apiErr.StatusCode != 0 || apiErr.Code != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeExceedContextSize {
		t.Errorf("APIError = %+v, want code 400 %s with zero StatusCode", apiErr, llama.ErrorTypeExceedContextSize)
	}
	if apiErr.NPromptTokens <= apiErr.NCtx || apiErr.NCtx == 0 {
		t.Errorf("context details = (%d, %d), want n_prompt_tokens > n_ctx > 0", apiErr.NPromptTokens, apiErr.NCtx)
	}
}

func TestCompletionsCheckFallbacks(t *testing.T) {
	jsontest.CheckFallbacks[CompletionRequest](t)
	jsontest.CheckFallbacks[CompletionResponse](t)
}

func TestCompletionsErrorWire(t *testing.T) {
	data := jsontest.ReadFile(t, "testdata/completions/error.resp.json")
	resp := &http.Response{
		StatusCode: http.StatusBadRequest,
		Header:     http.Header{"Content-Type": {"application/json; charset=utf-8"}},
		Body:       io.NopCloser(bytes.NewReader(data)),
	}
	apiErr := llama.ReadAPIError(resp)
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Errorf("StatusCode = %d, want %d", apiErr.StatusCode, http.StatusBadRequest)
	}
	if apiErr.Code != http.StatusBadRequest {
		t.Errorf("Code = %d, want %d", apiErr.Code, http.StatusBadRequest)
	}
	if apiErr.Type != llama.ErrorTypeInvalidRequest {
		t.Errorf("Type = %q, want %q", apiErr.Type, llama.ErrorTypeInvalidRequest)
	}
	if !strings.Contains(apiErr.Message, "prompt") {
		t.Errorf("Message = %q, want it to mention prompt", apiErr.Message)
	}
}

func TestCompletionsNewRequest(t *testing.T) {
	tests := map[string]struct {
		baseURL   string
		client    []llama.ClientOption
		opts      []llama.RequestOption
		req       *CompletionRequest
		wantPath  string
		wantAuth  string
		wantQuery url.Values
	}{
		"success: default base URL and bearer auth": {
			baseURL:   "http://127.0.0.1:8080",
			client:    []llama.ClientOption{llama.WithAPIKey("secret-key")},
			opts:      []llama.RequestOption{llama.WithModelQuery("tinyllama-2"), llama.WithAutoload(true)},
			req:       &CompletionRequest{Model: "tinyllama-2", Prompt: llama.TextPrompt("Hello"), MaxTokens: new(8)},
			wantPath:  "/v1/completions",
			wantAuth:  "Bearer secret-key",
			wantQuery: url.Values{"model": {"tinyllama-2"}, "autoload": {"true"}},
		},
		"success: api prefix": {
			baseURL:   "http://127.0.0.1:8080/llama",
			req:       &CompletionRequest{Prompt: llama.TextPrompt("Hi"), Temperature: new(0.0)},
			wantPath:  "/llama/v1/completions",
			wantQuery: url.Values{},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			lc, err := llama.New(tt.baseURL, tt.client...)
			if err != nil {
				t.Fatal(err)
			}
			oc := NewClient(lc)
			httpReq, err := oc.Llama().NewRequest(t.Context(), http.MethodPost, "/v1/completions", tt.req, tt.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if httpReq.Method != http.MethodPost {
				t.Errorf("Method = %q, want %q", httpReq.Method, http.MethodPost)
			}
			if httpReq.URL.Path != tt.wantPath {
				t.Errorf("URL.Path = %q, want %q", httpReq.URL.Path, tt.wantPath)
			}
			if ct := httpReq.Header.Get("Content-Type"); ct != "application/json" {
				t.Errorf("Content-Type = %q, want %q", ct, "application/json")
			}
			if diff := gocmp.Diff(tt.wantQuery, httpReq.URL.Query()); diff != "" {
				t.Errorf("query diff (-want +got):\n%s", diff)
			}
			if tt.wantAuth != "" {
				if auth := httpReq.Header.Get("Authorization"); auth != tt.wantAuth {
					t.Errorf("Authorization = %q, want %q", auth, tt.wantAuth)
				}
			}
			bodyBytes, err := io.ReadAll(httpReq.Body)
			if err != nil {
				t.Fatal(err)
			}
			var decoded CompletionRequest
			if err := json.Unmarshal(bodyBytes, &decoded); err != nil {
				t.Fatalf("unmarshaling request body: %v", err)
			}
			if diff := gocmp.Diff(tt.req, &decoded,
				gocmp.Comparer(llama.Prompt.Equal),
				gocmp.Comparer(llama.LogitBias.Equal),
				gocmp.Comparer(llama.StringOrSlice.Equal),
			); diff != "" {
				t.Errorf("decoded request body diff (-want +got):\n%s", diff)
			}
		})
	}
}

func ExampleClient_Completions() {
	lc, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		return
	}
	oc := NewClient(lc)

	res, err := oc.Completions(context.Background(), &CompletionRequest{
		Model:       "tinyllama-2",
		Prompt:      llama.TextPrompt("Once upon a time"),
		MaxTokens:   new(16),
		Temperature: new(0.0),
	})
	if err != nil {
		return
	}
	fmt.Println(res.Choices[0].Text)
}
