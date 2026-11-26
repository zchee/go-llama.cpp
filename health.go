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
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"time"

	backoff "github.com/cenkalti/backoff/v5"
)

// HealthResponse is the body of a successful GET /health.
type HealthResponse struct {
	// Status is "ok" once the model is loaded.
	Status string `json:"status"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// Health reports whether the server is ready (GET /health). The endpoint needs no API key. While the model is
// still loading the server answers 503, which Health returns as an *APIError with ErrorTypeUnavailable; Health
// never retries, so the caller sees the current state. Use WaitReady to wait for readiness.
func (c *Client) Health(ctx context.Context, opts ...RequestOption) (*HealthResponse, error) {
	opts = append(slices.Clip(opts), WithNoRetry())
	res, err := Call[HealthResponse](ctx, c, http.MethodGet, "/health", nil, opts...)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// WaitReady polls GET /health until the server reports ready, ctx is done, or the server answers with an error
// that waiting cannot fix (for example 404 from a wrong API prefix). Connection failures and 503 responses are
// waited out, so WaitReady can be called while the server process is still starting.
func (c *Client) WaitReady(ctx context.Context) error {
	b := &backoff.ExponentialBackOff{
		InitialInterval:     50 * time.Millisecond,
		RandomizationFactor: 0.2,
		Multiplier:          1.5,
		MaxInterval:         time.Second,
	}
	b.Reset()
	for {
		_, err := c.Health(ctx)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return fmt.Errorf("llama: waiting for the server: %w (last error: %w)", context.Cause(ctx), err)
		}
		if apiErr, ok := errors.AsType[*APIError](err); ok && apiErr.StatusCode != http.StatusServiceUnavailable {
			return err
		}
		timer := time.NewTimer(b.NextBackOff())
		select {
		case <-ctx.Done():
			timer.Stop()
			return fmt.Errorf("llama: waiting for the server: %w (last error: %w)", context.Cause(ctx), err)
		case <-timer.C:
		}
	}
}
