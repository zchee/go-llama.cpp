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
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
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
	resp, err := c.do(ctx, http.MethodGet, "/health", nil, "application/json", opts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var res HealthResponse
	if err := json.UnmarshalRead(resp.Body, &res, jsonOptions); err != nil {
		return nil, c.RedactError(fmt.Errorf("llama: decoding GET /health response: %w", err))
	}
	if res.Status != "ok" {
		return nil, c.RedactError(fmt.Errorf("llama: GET /health returned status %q, want ok", res.Status))
	}
	return &res, nil
}

// retryableHealthError distinguishes connection failures before headers from invalid health responses.
func retryableHealthError(err error) bool {
	if e, ok := errors.AsType[*APIError](err); ok {
		return e.StatusCode == http.StatusServiceUnavailable
	}
	u, ok := errors.AsType[*url.Error](err)
	if !ok {
		return false
	}
	_, network := errors.AsType[*net.OpError](u.Err)
	return network || errors.Is(u.Err, io.EOF) || errors.Is(u.Err, io.ErrUnexpectedEOF)
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
			return c.RedactError(fmt.Errorf("llama: waiting for the server: %w (last error: %w)", context.Cause(ctx), err))
		}
		if !retryableHealthError(err) {
			return err
		}
		timer := time.NewTimer(b.NextBackOff())
		select {
		case <-ctx.Done():
			timer.Stop()
			return c.RedactError(fmt.Errorf("llama: waiting for the server: %w (last error: %w)", context.Cause(ctx), err))
		case <-timer.C:
		}
	}
}
