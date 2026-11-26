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
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// DefaultUserAgent is the User-Agent header sent unless WithUserAgent replaces it.
const DefaultUserAgent = "go-llama.cpp"

// DefaultMaxEventSize is the largest server-sent event a stream accepts unless WithMaxEventSize replaces it. Final
// events of the Responses API repeat the whole output, so the limit is far above the 64 KiB that SSE parsers
// commonly default to.
const DefaultMaxEventSize = 16 << 20

// Client is a client for the llama.cpp HTTP server (llama-server).
//
// A Client is immutable after New returns and is safe for concurrent use.
type Client struct {
	base         *url.URL
	apiKey       string
	apiKeyHeader APIKeyHeader
	hc           *http.Client
	userAgent    string
	retry        RetryPolicy
	logger       *slog.Logger
	maxEventSize int
}

// New returns a client for the server at baseURL, for example "http://127.0.0.1:8080". A path in baseURL is the
// server's --api-prefix and is prepended to every endpoint path.
func New(baseURL string, opts ...ClientOption) (*Client, error) {
	base, err := parseBaseURL(baseURL)
	if err != nil {
		return nil, err
	}

	cfg := &clientConfig{
		retry:        DefaultRetryPolicy(),
		userAgent:    DefaultUserAgent,
		maxEventSize: DefaultMaxEventSize,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
	if cfg.apiKeyHeader != APIKeyHeaderAuthorization && cfg.apiKeyHeader != APIKeyHeaderXAPIKey {
		return nil, fmt.Errorf("llama: unknown APIKeyHeader %d", cfg.apiKeyHeader)
	}
	if cfg.maxEventSize <= 0 {
		cfg.maxEventSize = DefaultMaxEventSize
	}
	if cfg.logger == nil {
		cfg.logger = slog.New(slog.DiscardHandler)
	}

	hc, err := buildHTTPClient(cfg.httpClient, cfg.unixSocket)
	if err != nil {
		return nil, err
	}

	return &Client{
		base:         base,
		apiKey:       cfg.apiKey,
		apiKeyHeader: cfg.apiKeyHeader,
		hc:           hc,
		userAgent:    cfg.userAgent,
		retry:        cfg.retry,
		logger:       cfg.logger,
		maxEventSize: cfg.maxEventSize,
	}, nil
}

// parseBaseURL validates baseURL and strips a trailing slash from its path.
func parseBaseURL(baseURL string) (*url.URL, error) {
	if baseURL == "" {
		return nil, errors.New("llama: empty base URL")
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("llama: invalid base URL: %w", err)
	}
	switch u.Scheme {
	case "http", "https":
	default:
		return nil, fmt.Errorf("llama: base URL scheme must be http or https, got %q", u.Scheme)
	}
	if u.Host == "" {
		return nil, errors.New("llama: base URL has no host")
	}
	if u.User != nil {
		return nil, errors.New("llama: base URL must not carry credentials; use WithAPIKey")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return nil, errors.New("llama: base URL must not have a query or fragment")
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	u.RawPath = ""
	return u, nil
}

// buildHTTPClient returns the HTTP client to use. The default client has no overall Timeout because it would also
// cut streams that are still producing tokens.
func buildHTTPClient(hc *http.Client, unixSocket string) (*http.Client, error) {
	if hc == nil {
		hc = &http.Client{Transport: http.DefaultTransport.(*http.Transport).Clone()}
	}
	if unixSocket == "" {
		return hc, nil
	}

	var tr *http.Transport
	switch rt := hc.Transport.(type) {
	case nil:
		tr = http.DefaultTransport.(*http.Transport).Clone()
	case *http.Transport:
		tr = rt.Clone()
	default:
		return nil, fmt.Errorf("llama: WithUnixSocket needs an *http.Transport, the HTTP client has %T", hc.Transport)
	}
	var d net.Dialer
	tr.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
		return d.DialContext(ctx, "unix", unixSocket)
	}
	tr.Proxy = nil

	out := &http.Client{
		Transport:     tr,
		CheckRedirect: hc.CheckRedirect,
		Jar:           hc.Jar,
		Timeout:       hc.Timeout,
	}
	return out, nil
}

// HTTPClient returns the HTTP client the Client sends requests with. Packages that speak another protocol over the
// same connection settings, such as the realtime WebSocket API, use it together with NewRequest. The returned
// client must not be modified.
func (c *Client) HTTPClient() *http.Client {
	return c.hc
}

// String returns a description of the client that never contains the API key.
func (c *Client) String() string {
	return fmt.Sprintf("llama.Client{base: %q, apiKey: %t}", c.base.String(), c.apiKey != "")
}

// Format implements fmt.Formatter so that no verb, %#v included, prints the unexported fields, which hold the API
// key.
func (c *Client) Format(f fmt.State, _ rune) {
	_, _ = fmt.Fprint(f, c.String())
}

// LogValue implements slog.LogValuer so that logging a Client never records the API key.
func (c *Client) LogValue() slog.Value {
	return slog.StringValue(c.String())
}
