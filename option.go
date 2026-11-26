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
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// APIKeyHeader selects the request header that carries the API key.
type APIKeyHeader int

const (
	// APIKeyHeaderAuthorization sends the key as "Authorization: Bearer <key>". It is the default.
	APIKeyHeaderAuthorization APIKeyHeader = iota
	// APIKeyHeaderXAPIKey sends the key as "X-Api-Key: <key>", the header Anthropic clients use.
	APIKeyHeaderXAPIKey
)

// ClientOption configures a Client.
type ClientOption func(*clientConfig)

// clientConfig collects the client options before New validates them.
type clientConfig struct {
	apiKey       string
	apiKeyHeader APIKeyHeader
	httpClient   *http.Client
	unixSocket   string
	userAgent    string
	retry        RetryPolicy
	logger       *slog.Logger
	maxEventSize int
}

// WithAPIKey sets the API key the server was started with (--api-key). The key is never logged and never appears
// in error strings.
func WithAPIKey(key string) ClientOption {
	return func(c *clientConfig) { c.apiKey = key }
}

// WithAPIKeyHeader selects the header that carries the API key. The server accepts both forms.
func WithAPIKeyHeader(h APIKeyHeader) ClientOption {
	return func(c *clientConfig) { c.apiKeyHeader = h }
}

// WithHTTPClient sets the HTTP client used for every request, including the WebSocket handshake of the realtime
// API. Its Timeout should be zero: an overall timeout also cuts streams that are still producing tokens. Use
// context deadlines instead.
func WithHTTPClient(hc *http.Client) ClientOption {
	return func(c *clientConfig) { c.httpClient = hc }
}

// WithUnixSocket connects to a server listening on a UNIX domain socket (a --host value ending in ".sock"). The
// base URL passed to New still decides the scheme, the Host header and the API prefix, so "http://localhost" is
// the usual choice.
//
// WithUnixSocket clones the HTTP client's transport (http.DefaultTransport if none is set) and replaces its
// dialers. The effective transport must be an *http.Transport.
func WithUnixSocket(path string) ClientOption {
	return func(c *clientConfig) { c.unixSocket = path }
}

// WithUserAgent sets the User-Agent header.
func WithUserAgent(ua string) ClientOption {
	return func(c *clientConfig) { c.userAgent = ua }
}

// WithRetry replaces the retry policy. Use RetryPolicy{MaxAttempts: 1} to disable retries for the whole client.
func WithRetry(p RetryPolicy) ClientOption {
	return func(c *clientConfig) { c.retry = p }
}

// WithLogger sets the logger that receives one debug record per request (method, path, status, duration and
// attempts) and one per retry. Request bodies and the API key are never logged.
func WithLogger(l *slog.Logger) ClientOption {
	return func(c *clientConfig) { c.logger = l }
}

// WithMaxEventSize sets the largest server-sent event, in bytes, that a stream accepts. The default is
// DefaultMaxEventSize.
func WithMaxEventSize(n int) ClientOption {
	return func(c *clientConfig) { c.maxEventSize = n }
}

// RetryPolicy bounds how a request is retried.
//
// A request is retried when the server answers 503 with ErrorTypeUnavailable (it rejected the request before
// processing it, so any method may be resent), or when an idempotent request (GET, HEAD, OPTIONS, PUT, DELETE)
// fails before any response arrives. A stream is never retried once its response has been accepted.
type RetryPolicy struct {
	// MaxAttempts is the total number of attempts, including the first. Values below 2 disable retries.
	MaxAttempts int
	// InitialInterval is the wait before the first retry.
	InitialInterval time.Duration
	// MaxInterval caps the wait between two attempts.
	MaxInterval time.Duration
	// Multiplier grows the wait after each retry.
	Multiplier float64
	// RandomizationFactor spreads each wait over [wait*(1-f), wait*(1+f)].
	RandomizationFactor float64
	// MaxElapsedTime stops retrying once the next attempt would start later than this after the first one.
	// Zero means no limit besides MaxAttempts and the context.
	MaxElapsedTime time.Duration
}

// DefaultRetryPolicy returns the policy a Client uses unless WithRetry replaces it.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxAttempts:         4,
		InitialInterval:     250 * time.Millisecond,
		MaxInterval:         5 * time.Second,
		Multiplier:          2,
		RandomizationFactor: 0.5,
		MaxElapsedTime:      30 * time.Second,
	}
}

// RequestOption configures a single request.
type RequestOption func(*requestConfig)

// requestConfig collects the request options.
type requestConfig struct {
	query   url.Values
	header  http.Header
	noRetry bool
}

// newRequestConfig applies opts to an empty configuration.
func newRequestConfig(opts []RequestOption) *requestConfig {
	rc := &requestConfig{}
	for _, opt := range opts {
		if opt != nil {
			opt(rc)
		}
	}
	return rc
}

// WithQuery adds a query parameter. Repeated calls with the same key add more values.
func WithQuery(key, value string) RequestOption {
	return func(rc *requestConfig) {
		if rc.query == nil {
			rc.query = url.Values{}
		}
		rc.query.Add(key, value)
	}
}

// WithHeader sets a request header, replacing earlier values of the same header.
func WithHeader(key, value string) RequestOption {
	return func(rc *requestConfig) {
		if rc.header == nil {
			rc.header = http.Header{}
		}
		rc.header.Set(key, value)
	}
}

// WithModelQuery sets the "model" query parameter, which a server in router mode uses to route GET requests and
// WebSocket connections. POST requests are routed by the "model" member of the body instead.
func WithModelQuery(name string) RequestOption {
	return WithQuery("model", name)
}

// WithAutoload sets the "autoload" query parameter, which tells a server in router mode whether to load the
// requested model when it is not loaded yet.
func WithAutoload(autoload bool) RequestOption {
	return WithQuery("autoload", strconv.FormatBool(autoload))
}

// WithNoRetry limits this library to one attempt. net/http may still follow redirects or replay an idempotent
// request after a reused connection fails.
func WithNoRetry() RequestOption {
	return func(rc *requestConfig) { rc.noRetry = true }
}
