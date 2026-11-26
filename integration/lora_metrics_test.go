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
	"net/url"
	"os"
	"path/filepath"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

var presetLoRAMetricsAdapters = &serverPreset{
	name: "lora-moe",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "stories15M-MOE"),
			Args:  []string{"--ctx-size", "512", "--lora", harness.Model(t, "stories15M-MOE-lora")},
		}
	},
}

var presetLoRAMetricsEnabled = &serverPreset{
	name: "metrics-stories260K",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "stories260K"),
			Args:  []string{"--ctx-size", "512", "--metrics"},
		}
	},
}

func TestLoRAAdaptersReplaceGlobalScales(t *testing.T) {
	adapter := harness.Model(t, "stories15M-MOE-lora")
	srv := harness.StartT(t, harness.Options{
		Model: harness.Model(t, "stories15M-MOE"),
		Args:  []string{"--ctx-size", "512", "--lora", adapter + "," + adapter, "--lora-init-without-apply"},
	})
	c, _ := srv.Client(t)
	initial, err := c.LoRAAdapters(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != 2 || initial[0].ID != 0 || initial[1].ID != 1 {
		t.Fatalf("adapters = %+v, want two adapters ordered by ID", initial)
	}
	// Loading without applying leaves the reported scales unchanged in llama-server v1.1.0.
	if diff := gocmp.Diff([2]float64{1, 1}, [2]float64{initial[0].Scale, initial[1].Scale}); diff != "" {
		t.Errorf("initial scales mismatch (-want +got):\n%s", diff)
	}

	tests := map[string]struct {
		weights []llama.LoRAWeight
		want    [2]float64
	}{
		"success: omission disables an adapter": {weights: []llama.LoRAWeight{{ID: 1, Scale: 0.75}}, want: [2]float64{0, 0.75}},
		"success: explicit zero disables":       {weights: []llama.LoRAWeight{{ID: 0, Scale: 0.5}, {ID: 1, Scale: 0}}, want: [2]float64{0.5, 0}},
		"success: nil disables all adapters":    {want: [2]float64{0, 0}},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			// Make each update replace two active weights, regardless of subtest order.
			if _, err := c.SetLoRAAdapters(t.Context(), []llama.LoRAWeight{{ID: 0, Scale: 1}, {ID: 1, Scale: 1}}); err != nil {
				t.Fatal(err)
			}
			res, err := c.SetLoRAAdapters(t.Context(), tc.weights)
			if err != nil {
				t.Fatal(err)
			}
			if res == nil || !res.Success {
				t.Fatalf("update = %+v, want success", res)
			}
			adapters, err := c.LoRAAdapters(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if len(adapters) != 2 {
				t.Fatalf("adapters = %+v, want two adapters", adapters)
			}
			if diff := gocmp.Diff(tc.want, [2]float64{adapters[0].Scale, adapters[1].Scale}); diff != "" {
				t.Errorf("scales mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLoRAMetricsServerErrors(t *testing.T) {
	tests := map[string]struct {
		server *serverPreset
		call   func(*testing.T, *llama.Client) error
		status int
		typ    llama.ErrorType
	}{
		"error: set adapters requires an array": {
			server: presetLoRAMetricsAdapters,
			call: func(t *testing.T, c *llama.Client) error {
				_, err := llama.Call[llama.LoRASetAdaptersResponse](t.Context(), c, http.MethodPost, "/lora-adapters", jsontext.Value(`{"id":0,"scale":0.5}`))
				return err
			},
			status: http.StatusBadRequest, typ: llama.ErrorTypeInvalidRequest,
		},
		"error: metrics disabled": {
			server: presetStories260K,
			call: func(t *testing.T, c *llama.Client) error {
				_, err := c.Metrics(t.Context())
				return err
			},
			status: http.StatusNotImplemented, typ: llama.ErrorTypeNotSupported,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			c, obs := tc.server.server(t).Client(t)
			err := tc.call(t, c)
			apiErr, ok := errors.AsType[*llama.APIError](err)
			if !ok || apiErr.StatusCode != tc.status || apiErr.Code != tc.status || apiErr.Type != tc.typ {
				t.Fatalf("error = %v, want *llama.APIError with HTTP %d %s", err, tc.status, tc.typ)
			}
			if xs := obs.Exchanges(); len(xs) != 1 || !xs[0].BodyClosed() {
				t.Errorf("exchanges = %+v, want one closed response body", xs)
			}
		})
	}
}

func TestMetricsSpeculativeDecoding(t *testing.T) {
	model := harness.Model(t, "stories260K")
	srv := harness.StartT(t, harness.Options{
		Model: model,
		Args:  []string{"--ctx-size", "512", "--metrics", "--spec-type", "draft-simple", "--spec-draft-model", model},
	})
	c, _ := srv.Client(t)
	_, err := llama.Call[jsontext.Value](t.Context(), c, http.MethodPost, "/completion", jsontext.Value(`{"prompt":"Once upon a time there was a little girl","n_predict":24,"temperature":0,"stream":false}`))
	if err != nil {
		t.Fatal(err)
	}

	scrape, err := c.Metrics(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if count, ok := scrape.TokensPredictedTotal(); !ok || count != 24 {
		t.Errorf("generated tokens = (%g, %t), want (24, true)", count, ok)
	}
	if value, ok := scrape.PromptTokensPerSecond(); !ok || value <= 0 {
		t.Errorf("prompt throughput = (%g, %t), want positive throughput", value, ok)
	}
	accepted, ok := scrape.SpecDecodeAcceptedTokensTotal()
	positions := scrape.SpecDecodeAcceptedTokensPerPosition()
	var sum float64
	for _, count := range positions {
		sum += count
	}
	if !ok || accepted <= 0 || sum != accepted {
		t.Errorf("accepted tokens = (%g, %t), positions = %v (sum %g), want positive counts that agree", accepted, ok, positions, sum)
	}

	next, err := c.Metrics(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if value, ok := next.PromptTokensPerSecond(); !ok || value != 0 {
		t.Errorf("prompt throughput after scraping = (%g, %t), want (0, true)", value, ok)
	}
	if diff := gocmp.Diff(positions, next.SpecDecodeAcceptedTokensPerPosition()); diff != "" {
		t.Errorf("position counters changed across scrapes (-want +got):\n%s", diff)
	}
}

func TestMetricsRouter(t *testing.T) {
	dir := t.TempDir()
	if err := os.Symlink(harness.Model(t, "stories260K"), filepath.Join(dir, "stories260K.gguf")); err != nil {
		t.Fatal(err)
	}
	srv := harness.StartT(t, harness.Options{ModelsDir: dir, Args: []string{"--ctx-size", "512", "--metrics"}})
	c, obs := srv.Client(t)
	if _, err := c.Metrics(t.Context()); err == nil {
		t.Fatal("unrouted scrape succeeded, want HTTP 400")
	} else if apiErr, ok := errors.AsType[*llama.APIError](err); !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest {
		t.Fatalf("unrouted scrape error = %v, want HTTP 400 invalid_request_error", err)
	}

	opts := llama.WithModelQuery("stories260K")
	scrape, err := c.Metrics(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if count, ok := scrape.PromptTokensTotal(); !ok || count != 0 {
		t.Errorf("idle prompt tokens = (%g, %t), want (0, true)", count, ok)
	}
	raw, err := c.MetricsText(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if raw == "" || raw != scrape.Raw {
		t.Errorf("raw scrape = %d bytes, want same bytes as the parsed idle scrape (%d bytes)", len(raw), len(scrape.Raw))
	}
	got := obs.Exchanges()
	wantQuery := []url.Values{{}, {"model": {"stories260K"}}, {"model": {"stories260K"}}}
	if len(got) != len(wantQuery) {
		t.Fatalf("request count = %d, want %d", len(got), len(wantQuery))
	}
	for i, exchange := range got {
		if diff := gocmp.Diff(wantQuery[i], exchange.Query); diff != "" {
			t.Errorf("request %d query mismatch (-want +got):\n%s", i, diff)
		}
		if exchange.Method != http.MethodGet || exchange.Path != "/metrics" || !exchange.BodyClosed() || exchange.BytesRead() == 0 {
			t.Errorf("request %d = %s %s (closed: %t, read: %d), want GET /metrics with a read and closed body", i, exchange.Method, exchange.Path, exchange.BodyClosed(), exchange.BytesRead())
		}
	}
}
