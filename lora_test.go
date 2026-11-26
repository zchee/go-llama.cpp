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
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestLoRAAdaptersWire(t *testing.T) {
	tests := map[string]struct {
		fixture string
		want    []LoRAAdapter
	}{
		"success: no adapters": {fixture: "empty", want: []LoRAAdapter{}},
		"success: initially enabled": {
			fixture: "enabled",
			want:    []LoRAAdapter{{ID: 0, Path: "moe_shakespeare15M.gguf", Scale: 1}},
		},
		"success: loaded without applying": {
			fixture: "init_without_apply",
			want:    []LoRAAdapter{{ID: 0, Path: "moe_shakespeare15M.gguf", Scale: 1}},
		},
		"success: after setting scale": {
			fixture: "scaled",
			want:    []LoRAAdapter{{ID: 0, Path: "moe_shakespeare15M.gguf", Scale: 0.5}},
		},
		"success: adapter with an invocation sequence": {
			fixture: "activation",
			want: []LoRAAdapter{
				{ID: 0, Path: "moe_shakespeare15M.gguf", Scale: 1},
				{ID: 1, Path: "invocation.gguf", Scale: 1, TaskName: "fixture task", PromptPrefix: "fixture prefix", ALoRAInvocationString: new("'("), ALoRAInvocationTokens: []Token{42, 43}},
			},
		},
		"success: unlisted adapter disabled": {
			fixture: "activation_scaled",
			want: []LoRAAdapter{
				{ID: 0, Path: "moe_shakespeare15M.gguf", Scale: 0},
				{ID: 1, Path: "invocation.gguf", Scale: 0.75, TaskName: "fixture task", PromptPrefix: "fixture prefix", ALoRAInvocationString: new("'("), ALoRAInvocationTokens: []Token{42, 43}},
			},
		},
	}
	jsontest.CheckFallbacks[LoRAAdapter](t)
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/lora_adapters/"+tc.fixture+".resp.json")
			got := jsontest.RoundTrip[[]LoRAAdapter](t, data)
			if diff := gocmp.Diff(tc.want, *got); diff != "" {
				t.Errorf("adapter list mismatch (-want +got):\n%s", diff)
			}
			// The recorded empty list has no object into which a member can be injected.
			if len(tc.want) > 0 {
				jsontest.InjectUnknown[[]LoRAAdapter](t, data)
			}
			jsontest.NoUnknown[[]LoRAAdapter](t, data, nil)
		})
	}
}

func TestLoRASetAdaptersWire(t *testing.T) {
	tests := map[string]struct {
		fixture string
		weights []LoRAWeight
	}{
		"success: scale update":          {fixture: "set", weights: []LoRAWeight{{ID: 0, Scale: 0.5}}},
		"success: update second adapter": {fixture: "activation_set", weights: []LoRAWeight{{ID: 1, Scale: 0.75}}},
		"success: clear all":             {fixture: "clear", weights: []LoRAWeight{}},
	}
	jsontest.CheckFallbacks[LoRAWeight](t)
	jsontest.CheckFallbacks[LoRASetAdaptersResponse](t)
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			request := jsontest.ReadFile(t, "testdata/lora_adapters/"+tc.fixture+".req.json")
			got := jsontest.RoundTrip[[]LoRAWeight](t, request)
			if diff := gocmp.Diff(tc.weights, *got); diff != "" {
				t.Errorf("weights mismatch (-want +got):\n%s", diff)
			}
			// Clearing adapters sends an empty array, not an object with an Extra map.
			if len(tc.weights) > 0 {
				jsontest.InjectUnknown[[]LoRAWeight](t, request)
			}
			jsontest.NoUnknown[[]LoRAWeight](t, request, nil)

			response := jsontest.ReadFile(t, "testdata/lora_adapters/"+tc.fixture+".resp.json")
			if got := jsontest.RoundTrip[LoRASetAdaptersResponse](t, response); !got.Success {
				t.Errorf("Success = %t, want true", got.Success)
			}
			jsontest.InjectUnknown[LoRASetAdaptersResponse](t, response)
			jsontest.NoUnknown[LoRASetAdaptersResponse](t, response, nil)
		})
	}
}

func TestLoRAAdaptersRequest(t *testing.T) {
	tests := map[string]struct {
		method  string
		body    any
		opts    []RequestOption
		query   url.Values
		fixture string
	}{
		"success: list in router mode": {method: http.MethodGet, opts: []RequestOption{WithModelQuery("stories/MOE")}, query: url.Values{"model": {"stories/MOE"}}},
		"success: set":                 {method: http.MethodPost, body: []LoRAWeight{{ID: 0, Scale: 0.5}}, query: url.Values{}, fixture: "set"},
		"success: clear with nil":      {method: http.MethodPost, body: []LoRAWeight(nil), query: url.Values{}, fixture: "clear"},
	}
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), tc.method, loraAdaptersPath, tc.body, tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if req.Method != tc.method || req.URL.Path != "/lora-adapters" {
				t.Errorf("request = %s %s, want %s /lora-adapters", req.Method, req.URL.Path, tc.method)
			}
			if diff := gocmp.Diff(tc.query, req.URL.Query()); diff != "" {
				t.Errorf("request query mismatch (-want +got):\n%s", diff)
			}
			if tc.fixture == "" {
				if req.Body != nil {
					t.Error("GET request has a body")
				}
				return
			}
			if ct := req.Header.Get("Content-Type"); ct != "application/json" {
				t.Errorf("request Content-Type = %q, want application/json", ct)
			}
			defer req.Body.Close()
			got, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			want := jsontest.ReadFile(t, "testdata/lora_adapters/"+tc.fixture+".req.json")
			if diff := gocmp.Diff(string(want), string(got)); diff != "" {
				t.Errorf("request body mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLoRAAdaptersErrorWire(t *testing.T) {
	tests := map[string]struct {
		fixture string
		message string
	}{
		"error: body is not an array":      {fixture: "non_array", message: "Request body must be an array"},
		"error: router rejects array post": {fixture: "router_missing_model", message: "model name is missing from the request"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/lora_adapters/"+tc.fixture+".resp.json")
			err := ReadAPIError(&http.Response{StatusCode: http.StatusBadRequest, Body: io.NopCloser(bytes.NewReader(data))})
			if err.Code != http.StatusBadRequest || err.Type != ErrorTypeInvalidRequest || err.Message != tc.message {
				t.Errorf("decoded API error = %+v, want HTTP 400 invalid_request_error: %s", err, tc.message)
			}
		})
	}
}

func ExampleClient_LoRAAdapters() {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		return
	}
	defer c.HTTPClient().CloseIdleConnections()
	adapters, err := c.LoRAAdapters(context.Background())
	if err != nil {
		return
	}
	for _, adapter := range adapters {
		fmt.Printf("adapter %d: %s (scale %g)\n", adapter.ID, adapter.Path, adapter.Scale)
	}
}
