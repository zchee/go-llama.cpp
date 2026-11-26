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
	"context"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

const metricsPath = "/metrics"

// MetricsResponse is a Prometheus text scrape. Each scalar accessor returns false
// if its metric is absent or does not have exactly one unlabeled sample of the expected type.
type MetricsResponse struct {
	// Raw is the original Prometheus text.
	Raw string
	// Families contains all parsed metric families, including ones without typed accessors.
	Families map[string]*dto.MetricFamily
}

// Metrics scrapes and parses the metrics of a server started with --metrics.
// In router mode, select a model with WithModelQuery.
func (c *Client) Metrics(ctx context.Context, opts ...RequestOption) (*MetricsResponse, error) {
	raw, err := c.MetricsText(ctx, opts...)
	if err != nil {
		return nil, err
	}
	res, err := metricsParse(raw)
	return res, c.RedactError(err)
}

// MetricsText returns the original Prometheus text without parsing it.
// The server needs --metrics, and in router mode the request needs WithModelQuery.
func (c *Client) MetricsText(ctx context.Context, opts ...RequestOption) (string, error) {
	resp, err := CallRaw(ctx, c, http.MethodGet, metricsPath, nil, opts...)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", c.RedactError(fmt.Errorf("llama: reading metrics response: %w", err))
	}
	return string(data), nil
}

func metricsParse(raw string) (*MetricsResponse, error) {
	parser := expfmt.NewTextParser(model.LegacyValidation)
	families, err := parser.TextToMetricFamilies(strings.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("llama: decoding metrics response: %w", err)
	}
	return &MetricsResponse{Raw: raw, Families: families}, nil
}

// PromptTokensTotal returns the number of prompt tokens processed, excluding cached tokens.
func (m *MetricsResponse) PromptTokensTotal() (float64, bool) {
	return m.metricsCounter("prompt_tokens_total")
}

// PromptTokensCachedTotal returns the number of prompt tokens reused from the cache.
func (m *MetricsResponse) PromptTokensCachedTotal() (float64, bool) {
	return m.metricsCounter("prompt_tokens_cached_total")
}

// PromptSecondsTotal returns the total time spent processing prompts in seconds.
func (m *MetricsResponse) PromptSecondsTotal() (float64, bool) {
	return m.metricsCounter("prompt_seconds_total")
}

// PromptTokensPerSecond returns average prompt processing throughput in tokens per second.
func (m *MetricsResponse) PromptTokensPerSecond() (float64, bool) {
	return m.metricsGauge("prompt_tokens_seconds")
}

// TokensPredictedTotal returns the number of generated tokens.
func (m *MetricsResponse) TokensPredictedTotal() (float64, bool) {
	return m.metricsCounter("tokens_predicted_total")
}

// TokensPredictedSecondsTotal returns the total time spent generating tokens in seconds.
func (m *MetricsResponse) TokensPredictedSecondsTotal() (float64, bool) {
	return m.metricsCounter("tokens_predicted_seconds_total")
}

// PredictedTokensPerSecond returns average generation throughput in tokens per second.
func (m *MetricsResponse) PredictedTokensPerSecond() (float64, bool) {
	return m.metricsGauge("predicted_tokens_seconds")
}

// RequestsProcessing returns the number of requests being processed.
func (m *MetricsResponse) RequestsProcessing() (float64, bool) {
	return m.metricsGauge("requests_processing")
}

// RequestsDeferred returns the number of deferred requests.
func (m *MetricsResponse) RequestsDeferred() (float64, bool) {
	return m.metricsGauge("requests_deferred")
}

// TokensMax returns the largest observed sequence length, including prompt and generated tokens.
func (m *MetricsResponse) TokensMax() (float64, bool) { return m.metricsCounter("n_tokens_max") }

// DecodeTotal returns the number of llama_decode calls, excluding speculative and multimodal decoding.
func (m *MetricsResponse) DecodeTotal() (float64, bool) { return m.metricsCounter("n_decode_total") }

// BusySlotsPerDecode returns the average number of busy slots per llama_decode call.
func (m *MetricsResponse) BusySlotsPerDecode() (float64, bool) {
	return m.metricsGauge("n_busy_slots_per_decode")
}

// SpecDecodeDraftTokensTotal returns the number of tokens drafted by speculative decoding.
func (m *MetricsResponse) SpecDecodeDraftTokensTotal() (float64, bool) {
	return m.metricsCounter("spec_decode_num_draft_tokens_total")
}

// SpecDecodeAcceptedTokensTotal returns the number of draft tokens accepted by the target model.
func (m *MetricsResponse) SpecDecodeAcceptedTokensTotal() (float64, bool) {
	return m.metricsCounter("spec_decode_num_accepted_tokens_total")
}

// SpecDecodeDraftsTotal returns the number of speculative decoding verification steps.
func (m *MetricsResponse) SpecDecodeDraftsTotal() (float64, bool) {
	return m.metricsCounter("spec_decode_num_drafts_total")
}

// SpecDecodeAcceptedTokensPerPosition returns accepted draft token counts keyed by zero-based draft
// position, or nil when there are no such samples.
func (m *MetricsResponse) SpecDecodeAcceptedTokensPerPosition() map[int]float64 {
	family := m.Families["llamacpp:spec_decode_num_accepted_tokens_per_pos_total"]
	if family.GetType() != dto.MetricType_COUNTER {
		return nil
	}
	var out map[int]float64
	for _, sample := range family.GetMetric() {
		labels := sample.GetLabel()
		counter := sample.GetCounter()
		if len(labels) != 1 || labels[0].GetName() != "position" || counter == nil || counter.Value == nil {
			continue
		}
		pos, err := strconv.Atoi(labels[0].GetValue())
		if err != nil || pos < 0 {
			continue
		}
		if out == nil {
			out = make(map[int]float64, len(family.GetMetric()))
		}
		out[pos] = counter.GetValue()
	}
	return out
}

func (m *MetricsResponse) metricsCounter(name string) (float64, bool) {
	counter := m.metricsSample(name, dto.MetricType_COUNTER).GetCounter()
	return counter.GetValue(), counter != nil && counter.Value != nil
}

func (m *MetricsResponse) metricsGauge(name string) (float64, bool) {
	gauge := m.metricsSample(name, dto.MetricType_GAUGE).GetGauge()
	return gauge.GetValue(), gauge != nil && gauge.Value != nil
}

func (m *MetricsResponse) metricsSample(name string, typ dto.MetricType) *dto.Metric {
	family := m.Families["llamacpp:"+name]
	samples := family.GetMetric()
	if family.GetType() != typ || len(samples) != 1 || len(samples[0].GetLabel()) != 0 {
		return nil
	}
	return samples[0]
}
