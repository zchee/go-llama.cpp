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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	backoff "github.com/cenkalti/backoff/v5"

	"github.com/zchee/go-llama.cpp/internal/extraguard"
)

// Body is a request body that is sent as is, for payloads that are not JSON such as multipart/form-data.
//
// Data is kept in memory so that a retried request resends exactly the same bytes.
type Body struct {
	// ContentType is the value of the Content-Type header.
	ContentType string
	// Data is the encoded body.
	Data []byte
}

// encodeBody converts the body argument of NewRequest, Call, Stream and CallRaw into bytes and a content type.
func encodeBody(body any) ([]byte, string, error) {
	switch b := body.(type) {
	case nil:
		return nil, "", nil
	case *Body:
		if b == nil {
			return nil, "", nil
		}
		return bytes.Clone(b.Data), b.ContentType, nil
	case Body:
		return bytes.Clone(b.Data), b.ContentType, nil
	case jsontext.Value:
		return b.Clone(), "application/json", nil
	default:
		if err := extraguard.Check(body); err != nil {
			return nil, "", err
		}
		data, err := json.Marshal(body, jsonOptions)
		if err != nil {
			return nil, "", fmt.Errorf("llama: encoding request body: %w", err)
		}
		return data, "application/json", nil
	}
}

// NewRequest builds the HTTP request for an endpoint without sending it. The URL is the base URL followed by path,
// which is an unescaped path that starts with "/" and may carry a query string; the request carries the
// authentication and User-Agent headers of the client.
//
// The body is encoded as follows: nil sends no body, *Body and Body are sent as is, jsontext.Value is sent as
// JSON, and any other value is marshaled with encoding/json/v2. The body is held in memory and GetBody is set, so
// the request can be resent. In a struct body, a key in an Extra fallback map must not be the JSON name of a
// field in the same object, even when the typed field is omitted. Set the typed field instead.
func (c *Client) NewRequest(ctx context.Context, method, path string, body any, opts ...RequestOption) (*http.Request, error) {
	req, _, err := c.buildRequest(ctx, method, path, body, "", opts)
	return req, c.RedactError(err)
}

// buildRequest is NewRequest with a default Accept header, also returning the request options.
func (c *Client) buildRequest(ctx context.Context, method, path string, body any, accept string, opts []RequestOption) (*http.Request, *requestConfig, error) {
	data, contentType, err := encodeBody(body)
	if err != nil {
		return nil, nil, err
	}
	rc := newRequestConfig(opts)
	u, err := c.resolve(path, rc.query)
	if err != nil {
		return nil, nil, err
	}

	var rd io.Reader
	if data != nil {
		// A *bytes.Reader makes net/http set ContentLength and a GetBody that replays the same bytes.
		rd = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, u.String(), rd)
	if err != nil {
		return nil, nil, fmt.Errorf("llama: building request: %w", err)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	if c.userAgent != "" {
		req.Header.Set("User-Agent", c.userAgent)
	}
	if key := c.apiKey.value(); key != "" {
		switch c.apiKeyHeader {
		case APIKeyHeaderXAPIKey:
			req.Header.Set("X-Api-Key", key)
		default:
			req.Header.Set("Authorization", "Bearer "+key)
		}
	}
	maps.Copy(req.Header, rc.header)
	return req, rc, nil
}

// resolve joins the base URL and an endpoint path, merging the query string of path with query.
func (c *Client) resolve(path string, query url.Values) (*url.URL, error) {
	p, rawQuery, _ := strings.Cut(path, "?")
	if !strings.HasPrefix(p, "/") {
		return nil, fmt.Errorf("llama: endpoint path %q must start with /", path)
	}
	vals, err := url.ParseQuery(rawQuery)
	if err != nil {
		return nil, fmt.Errorf("llama: invalid query in endpoint path %q: %w", path, err)
	}
	for k, vs := range query {
		vals[k] = append(vals[k], vs...)
	}
	u := *c.base
	u.Path = c.base.Path + p
	u.RawPath = ""
	u.RawQuery = vals.Encode()
	return &u, nil
}

// do sends a request, retrying it under the client's policy, and returns a response with a 2xx status. Any other
// status is returned as an *APIError with the body already closed. Every error it returns, and every log record it
// writes, has the API key removed: transports and proxies may quote request headers in their errors.
func (c *Client) do(ctx context.Context, method, path string, body any, accept string, opts []RequestOption) (*http.Response, error) {
	req, rc, err := c.buildRequest(ctx, method, path, body, accept, opts)
	if err != nil {
		return nil, c.RedactError(err)
	}
	start := time.Now()
	policy := c.retry
	if rc.noRetry {
		policy.MaxAttempts = 1
	}
	attempts := 0
	resp, err := retry(ctx, policy, func(ctx context.Context, n int) (*http.Response, error) {
		attempts = n
		return c.attempt(ctx, req, n)
	})
	err = c.RedactError(err)

	attrs := []slog.Attr{
		slog.String("method", c.apiKey.redact(req.Method)),
		slog.String("path", c.apiKey.redact(req.URL.Path)),
		slog.Int("attempts", attempts),
		slog.Duration("duration", time.Since(start)),
	}
	if resp != nil {
		attrs = append(attrs, slog.Int("status", resp.StatusCode))
	} else if apiErr, ok := errors.AsType[*APIError](err); ok {
		attrs = append(attrs, slog.Int("status", apiErr.StatusCode))
	}
	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	c.logger.LogAttrs(ctx, slog.LevelDebug, "llama request", attrs...)
	return resp, err
}

// attempt sends attempt n of req and classifies the outcome for retry.
func (c *Client) attempt(ctx context.Context, req *http.Request, n int) (*http.Response, error) {
	r, err := requestForAttempt(req, n)
	if err != nil {
		return nil, err
	}
	if n > 1 {
		c.logger.LogAttrs(ctx, slog.LevelDebug, "llama retry",
			slog.String("method", c.apiKey.redact(req.Method)), slog.String("path", c.apiKey.redact(req.URL.Path)), slog.Int("attempt", n))
	}
	resp, err := c.hc.Do(r)
	if err != nil {
		if retryableTransportError(ctx, req.Method, err) {
			return nil, &retryableError{err: err}
		}
		return nil, err
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	apiErr := ReadAPIError(resp)
	if retryableStatus(apiErr) {
		// The slots handler uses the same unavailable_error when asked not to wait for a free slot.
		if req.Method == http.MethodGet && req.URL.Path == c.base.Path+"/slots" {
			values := req.URL.Query()["fail_on_no_slot"]
			if len(values) > 0 && values[len(values)-1] != "" {
				return nil, apiErr
			}
		}
		return nil, &retryableError{err: apiErr, after: retryAfter(resp.Header)}
	}
	return nil, apiErr
}

// requestForAttempt returns the request to send as attempt n. The first attempt sends req itself; later attempts
// send a clone whose body is regenerated with GetBody, so every attempt sends the same bytes.
func requestForAttempt(req *http.Request, n int) (*http.Request, error) {
	if n <= 1 {
		return req, nil
	}
	r := req.Clone(req.Context())
	if req.GetBody != nil {
		body, err := req.GetBody()
		if err != nil {
			return nil, fmt.Errorf("llama: regenerating request body: %w", err)
		}
		r.Body = body
	} else if req.Body != nil && req.Body != http.NoBody {
		return nil, errors.New("llama: request body cannot be resent")
	}
	return r, nil
}

// retryableTransportError reports whether a transport error may be retried: only for idempotent methods, because
// the server may have received a non-idempotent request before the connection failed, and never once ctx is done.
func retryableTransportError(ctx context.Context, method string, err error) bool {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	switch method {
	case http.MethodGet, http.MethodHead, http.MethodOptions, http.MethodPut, http.MethodDelete:
		return true
	default:
		return false
	}
}

// retryableStatus reports whether an error response may be retried. The server answers 503 unavailable_error
// before it starts processing a request (model loading, router child starting), so any method may be resent.
func retryableStatus(e *APIError) bool {
	return e.StatusCode == http.StatusServiceUnavailable && e.Type == ErrorTypeUnavailable
}

// retryAfter parses a Retry-After header given in seconds.
func retryAfter(h http.Header) time.Duration {
	s := h.Get("Retry-After")
	if s == "" {
		return 0
	}
	sec, err := strconv.Atoi(s)
	if err != nil || sec <= 0 {
		return 0
	}
	return time.Duration(sec) * time.Second
}

// retryableError marks an attempt error that retry may retry.
type retryableError struct {
	err   error
	after time.Duration
}

func (e *retryableError) Error() string { return e.err.Error() }

func (e *retryableError) Unwrap() error { return e.err }

// retry calls attempt until it succeeds, it returns an error that is not a *retryableError, the policy's limits
// are reached, or ctx is done. n is the 1-based attempt number. The returned error is the last attempt's error
// without the retryable mark; when ctx ends the wait, it is the context's cause together with that error.
func retry[T any](ctx context.Context, p RetryPolicy, attempt func(ctx context.Context, n int) (T, error)) (T, error) {
	if p.MaxAttempts < 2 {
		v, err := attempt(ctx, 1)
		if re, ok := errors.AsType[*retryableError](err); ok {
			return v, re.err
		}
		return v, err
	}

	var (
		n         int
		last      error
		permanent bool
	)
	op := func() (T, error) {
		n++
		v, err := attempt(ctx, n)
		if err == nil {
			return v, nil
		}
		re, ok := errors.AsType[*retryableError](err)
		if !ok {
			permanent = true
			return v, backoff.Permanent(err)
		}
		last = re.err
		if re.after > 0 {
			wait := re.after
			if p.MaxInterval > 0 {
				wait = min(wait, p.MaxInterval)
			}
			return v, &backoff.RetryAfterError{Duration: wait}
		}
		return v, err
	}

	b := &backoff.ExponentialBackOff{
		InitialInterval:     p.InitialInterval,
		RandomizationFactor: min(max(p.RandomizationFactor, 0), 1),
		Multiplier:          max(p.Multiplier, 1),
		MaxInterval:         p.MaxInterval,
	}
	if b.MaxInterval <= 0 {
		b.MaxInterval = backoff.DefaultMaxInterval
	}
	v, err := backoff.Retry(ctx, op,
		backoff.WithBackOff(&cappedBackOff{BackOff: b, limit: b.MaxInterval}),
		backoff.WithMaxTries(uint(p.MaxAttempts)),
		backoff.WithMaxElapsedTime(p.MaxElapsedTime),
	)
	switch {
	case err == nil, permanent:
		return v, err
	case last == nil:
		return v, err
	}
	if _, ok := errors.AsType[*retryableError](err); ok {
		return v, last
	}
	if _, ok := errors.AsType[*backoff.RetryAfterError](err); ok {
		return v, last
	}
	// The context ended while waiting for the next attempt.
	return v, fmt.Errorf("%w (last attempt: %w)", err, last)
}

// cappedBackOff limits every wait of a BackOff. ExponentialBackOff caps only the interval before randomization, and
// uses an InitialInterval above MaxInterval unchanged, so its waits can exceed MaxInterval.
type cappedBackOff struct {
	backoff.BackOff
	limit time.Duration
}

func (b *cappedBackOff) NextBackOff() time.Duration {
	d := b.BackOff.NextBackOff()
	if d == backoff.Stop {
		return d
	}
	return min(d, b.limit)
}

// Call sends a request and decodes the JSON response body into Resp. It is the building block of the typed
// methods and reaches any route, including the documented aliases and undocumented routes.
//
// See NewRequest for how body is encoded. A response with a non-2xx status is returned as an *APIError. An empty
// response body leaves Resp at its zero value. Any error returns the zero value of Resp, never a partial decode.
func Call[Resp any](ctx context.Context, c *Client, method, path string, body any, opts ...RequestOption) (Resp, error) {
	var out Resp
	resp, err := c.do(ctx, method, path, body, "application/json", opts)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return out, c.RedactError(fmt.Errorf("llama: reading %s %s response: %w", method, path, err))
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return out, nil
	}
	if err := json.Unmarshal(data, &out, jsonOptions); err != nil {
		var zero Resp
		return zero, c.RedactError(fmt.Errorf("llama: decoding %s %s response: %w", method, path, err))
	}
	return out, nil
}

// CallRaw sends a request and returns the response with a 2xx status without decoding its body. The caller must
// close the Body. When an API key is configured, the Body is wrapped to redact errors from Read, Close, and WriteTo
// if supported by the original Body. A non-2xx status is returned as an *APIError and the body is closed.
// CallRaw serves endpoints whose responses are not JSON, such as audio and Prometheus metrics.
func CallRaw(ctx context.Context, c *Client, method, path string, body any, opts ...RequestOption) (*http.Response, error) {
	resp, err := c.do(ctx, method, path, body, "", opts)
	if err == nil && c.apiKey != nil {
		b := redactedBody{resp.Body, c}
		if wt, ok := resp.Body.(io.WriterTo); ok {
			resp.Body = &redactedWriterToBody{redactedBody: b, writerTo: wt}
		} else {
			resp.Body = &b
		}
	}
	return resp, err
}

type redactedBody struct {
	io.ReadCloser
	client *Client
}

func (b *redactedBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err == io.EOF {
		return n, io.EOF
	}
	return n, b.client.RedactError(err)
}

func (b *redactedBody) Close() error { return b.client.RedactError(b.ReadCloser.Close()) }

type redactedWriterToBody struct {
	redactedBody
	writerTo io.WriterTo
}

func (b *redactedWriterToBody) WriteTo(w io.Writer) (int64, error) {
	n, err := b.writerTo.WriteTo(w)
	return n, b.client.RedactError(err)
}
