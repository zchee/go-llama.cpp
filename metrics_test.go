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
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	dto "github.com/prometheus/client_model/go"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestMetricsWire(t *testing.T) {
	tests := map[string]struct {
		fixture   string
		positions map[int]float64
	}{
		"success: idle server":        {fixture: "idle"},
		"success: router scrape":      {fixture: "routed"},
		"success: speculative decode": {fixture: "speculative", positions: map[int]float64{0: 6, 1: 6, 2: 5}},
	}
	scalars := map[string]struct {
		accessor    func(*MetricsResponse) (float64, bool)
		speculative float64
	}{
		"prompt_tokens_total":                   {(*MetricsResponse).PromptTokensTotal, 12},
		"prompt_tokens_cached_total":            {(*MetricsResponse).PromptTokensCachedTotal, 0},
		"prompt_seconds_total":                  {(*MetricsResponse).PromptSecondsTotal, 0.003236},
		"prompt_tokens_seconds":                 {(*MetricsResponse).PromptTokensPerSecond, 3708.28},
		"tokens_predicted_total":                {(*MetricsResponse).TokensPredictedTotal, 24},
		"tokens_predicted_seconds_total":        {(*MetricsResponse).TokensPredictedSecondsTotal, 0.039027},
		"predicted_tokens_seconds":              {(*MetricsResponse).PredictedTokensPerSecond, 589.336},
		"requests_processing":                   {(*MetricsResponse).RequestsProcessing, 0},
		"requests_deferred":                     {(*MetricsResponse).RequestsDeferred, 0},
		"n_tokens_max":                          {(*MetricsResponse).TokensMax, 35},
		"n_decode_total":                        {(*MetricsResponse).DecodeTotal, 7},
		"n_busy_slots_per_decode":               {(*MetricsResponse).BusySlotsPerDecode, 1},
		"spec_decode_num_draft_tokens_total":    {(*MetricsResponse).SpecDecodeDraftTokensTotal, 17},
		"spec_decode_num_accepted_tokens_total": {(*MetricsResponse).SpecDecodeAcceptedTokensTotal, 17},
		"spec_decode_num_drafts_total":          {(*MetricsResponse).SpecDecodeDraftsTotal, 6},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			raw := string(jsontest.ReadFile(t, "testdata/metrics/"+tc.fixture+".resp.txt"))
			res, err := metricsParse(raw)
			if err != nil {
				t.Fatal(err)
			}
			if res.Raw != raw {
				t.Error("parsed response did not preserve the original raw text")
			}
			for metric, scalar := range scalars {
				if family := res.Families["llamacpp:"+metric]; family.GetName() != "llamacpp:"+metric {
					t.Errorf("Families missing llamacpp:%s", metric)
				}
				want := 0.0
				if tc.positions != nil {
					want = scalar.speculative
				}
				if got, ok := scalar.accessor(res); !ok || got != want {
					t.Errorf("accessor for %s = (%g, %t), want (%g, true)", metric, got, ok, want)
				}
			}
			if diff := gocmp.Diff(tc.positions, res.SpecDecodeAcceptedTokensPerPosition()); diff != "" {
				t.Errorf("accepted tokens per position mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMetricsRequest(t *testing.T) {
	tests := map[string]struct {
		opts []RequestOption
		want url.Values
	}{
		"success: single model": {want: url.Values{}},
		"success: router":       {opts: []RequestOption{WithModelQuery("org/stories:Q4"), WithAutoload(false)}, want: url.Values{"model": {"org/stories:Q4"}, "autoload": {"false"}}},
	}
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodGet, metricsPath, nil, tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if req.Method != http.MethodGet || req.URL.Path != "/metrics" || req.Body != nil {
				t.Errorf("request = %s %s (body: %v), want GET /metrics without a body", req.Method, req.URL.Path, req.Body != nil)
			}
			if diff := gocmp.Diff(tc.want, req.URL.Query()); diff != "" {
				t.Errorf("request query mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMetricsErrorWire(t *testing.T) {
	tests := map[string]struct {
		fixture string
		status  int
		typ     ErrorType
		message string
	}{
		"error: metrics disabled": {
			fixture: "disabled", status: http.StatusNotImplemented, typ: ErrorTypeNotSupported,
			message: "This server does not support metrics endpoint. Start it with `--metrics`",
		},
		"error: router without a model": {
			fixture: "router_missing_model", status: http.StatusBadRequest, typ: ErrorTypeInvalidRequest,
			message: "model name is missing from the request",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/metrics/"+tc.fixture+".resp.json")
			err := ReadAPIError(&http.Response{StatusCode: tc.status, Body: io.NopCloser(bytes.NewReader(data))})
			if err.Code != tc.status || err.Type != tc.typ || err.Message != tc.message {
				t.Errorf("decoded API error = %+v, want HTTP %d %s: %s", err, tc.status, tc.typ, tc.message)
			}
			if _, err := metricsParse(string(data)); err == nil || !strings.Contains(err.Error(), "decoding metrics response") {
				t.Errorf("metricsParse error = %v, want a metrics decode error", err)
			}
		})
	}
}

func TestMetricsCodecEdgeCases(t *testing.T) {
	tests := map[string]struct {
		raw   string
		check func(*testing.T, *MetricsResponse)
	}{
		"success: non-canonical position labels ignored in accessor and kept in Families": {
			raw: strings.Join([]string{
				"# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter",
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 6`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="00"} 99`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="+1"} 88`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="02"} 77`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="-1"} 55`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="abc"} 33`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 4`,
				"",
			}, "\n"),
			check: func(t *testing.T, res *MetricsResponse) {
				want := map[int]float64{0: 6, 1: 4}
				if diff := gocmp.Diff(want, res.SpecDecodeAcceptedTokensPerPosition()); diff != "" {
					t.Errorf("accepted tokens per position mismatch (-want +got):\n%s", diff)
				}
				family := res.Families["llamacpp:spec_decode_num_accepted_tokens_per_pos_total"]
				if got := len(family.GetMetric()); got != 7 {
					t.Errorf("Families sample count = %d, want 7", got)
				}
			},
		},
		"success: only non-canonical position labels returns nil": {
			raw: strings.Join([]string{
				"# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter",
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="00"} 9`,
				`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="+0"} 8`,
				"",
			}, "\n"),
			check: func(t *testing.T, res *MetricsResponse) {
				if got := res.SpecDecodeAcceptedTokensPerPosition(); got != nil {
					t.Errorf("SpecDecodeAcceptedTokensPerPosition() = %v, want nil", got)
				}
				family := res.Families["llamacpp:spec_decode_num_accepted_tokens_per_pos_total"]
				if got := len(family.GetMetric()); got != 2 {
					t.Errorf("Families sample count = %d, want 2", got)
				}
			},
		},
		"success: unknown family preserved": {
			raw: strings.Join([]string{
				"# HELP custom_slot_metric Custom metric",
				"# TYPE custom_slot_metric gauge",
				`custom_slot_metric{slot="0"} 42`,
				`custom_slot_metric{slot="1"} 17`,
				"",
			}, "\n"),
			check: func(t *testing.T, res *MetricsResponse) {
				if len(res.Families) != 1 {
					t.Fatalf("len(Families) = %d, want 1", len(res.Families))
				}
				family := res.Families["custom_slot_metric"]
				if family.GetType() != dto.MetricType_GAUGE || len(family.GetMetric()) != 2 {
					t.Fatalf("unknown family = %+v, want GAUGE with 2 samples", family)
				}
				if got, ok := res.PromptTokensTotal(); ok || got != 0 {
					t.Errorf("PromptTokensTotal() = (%g, %t), want (0, false)", got, ok)
				}
			},
		},
		"success: empty scrape": {
			raw: "",
			check: func(t *testing.T, res *MetricsResponse) {
				if len(res.Families) != 0 {
					t.Errorf("len(Families) = %d, want 0", len(res.Families))
				}
				if got, ok := res.PromptTokensTotal(); ok || got != 0 {
					t.Errorf("PromptTokensTotal() = (%g, %t), want (0, false)", got, ok)
				}
				if got, ok := res.PromptTokensPerSecond(); ok || got != 0 {
					t.Errorf("PromptTokensPerSecond() = (%g, %t), want (0, false)", got, ok)
				}
				if got := res.SpecDecodeAcceptedTokensPerPosition(); got != nil {
					t.Errorf("SpecDecodeAcceptedTokensPerPosition() = %v, want nil", got)
				}
			},
		},
		"success: comments only scrape": {
			raw: "# HELP llamacpp:prompt_tokens_total idle header without samples\n# TYPE llamacpp:prompt_tokens_total counter\n",
			check: func(t *testing.T, res *MetricsResponse) {
				if got, ok := res.PromptTokensTotal(); ok || got != 0 {
					t.Errorf("PromptTokensTotal() = (%g, %t), want (0, false)", got, ok)
				}
			},
		},
		"success: multi-sample scalar family rejected by accessor and kept in Families": {
			raw: strings.Join([]string{
				"# TYPE llamacpp:prompt_tokens_total counter",
				`llamacpp:prompt_tokens_total{slot="0"} 5`,
				`llamacpp:prompt_tokens_total{slot="1"} 7`,
				"# TYPE llamacpp:predicted_tokens_seconds gauge",
				`llamacpp:predicted_tokens_seconds{slot="0"} 12.5`,
				"",
			}, "\n"),
			check: func(t *testing.T, res *MetricsResponse) {
				if got, ok := res.PromptTokensTotal(); ok || got != 0 {
					t.Errorf("PromptTokensTotal() = (%g, %t), want (0, false)", got, ok)
				}
				if got, ok := res.PredictedTokensPerSecond(); ok || got != 0 {
					t.Errorf("PredictedTokensPerSecond() = (%g, %t), want (0, false)", got, ok)
				}
				if got := len(res.Families["llamacpp:prompt_tokens_total"].GetMetric()); got != 2 {
					t.Errorf("Families prompt_tokens_total samples = %d, want 2", got)
				}
			},
		},
		"success: wrong metric type rejected by accessor": {
			raw: strings.Join([]string{
				"# TYPE llamacpp:prompt_tokens_total gauge",
				"llamacpp:prompt_tokens_total 5",
				"",
			}, "\n"),
			check: func(t *testing.T, res *MetricsResponse) {
				if got, ok := res.PromptTokensTotal(); ok || got != 0 {
					t.Errorf("PromptTokensTotal() = (%g, %t), want (0, false)", got, ok)
				}
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			res, err := metricsParse(tc.raw)
			if err != nil {
				t.Fatalf("metricsParse() error = %v", err)
			}
			if res.Raw != tc.raw {
				t.Errorf("res.Raw = %q, want %q", res.Raw, tc.raw)
			}
			tc.check(t, res)
		})
	}
}

func ExampleClient_Metrics() {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		return
	}
	defer c.HTTPClient().CloseIdleConnections()
	scrape, err := c.Metrics(context.Background())
	if err != nil {
		return
	}
	if count, ok := scrape.TokensPredictedTotal(); ok {
		fmt.Printf("generated tokens: %g\n", count)
	}
}
