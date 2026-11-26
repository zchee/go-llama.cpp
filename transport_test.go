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
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	gocmp "github.com/google/go-cmp/cmp"
)

func TestNewRequest(t *testing.T) {
	type payload struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	tests := map[string]struct {
		baseURL    string
		opts       []ClientOption
		method     string
		path       string
		body       any
		reqOpts    []RequestOption
		wantURL    string
		wantHeader http.Header
		wantBody   string
	}{
		"success: GET without body or key": {
			baseURL: "http://127.0.0.1:8080",
			method:  http.MethodGet,
			path:    "/health",
			wantURL: "http://127.0.0.1:8080/health",
			wantHeader: http.Header{
				"User-Agent": {DefaultUserAgent},
			},
		},
		"success: API prefix and JSON body with bearer key": {
			baseURL: "http://127.0.0.1:8080/api/v2/",
			opts:    []ClientOption{WithAPIKey(testAPIKey)},
			method:  http.MethodPost,
			path:    "/v1/chat/completions",
			body:    &payload{Model: "m", Stream: true},
			wantURL: "http://127.0.0.1:8080/api/v2/v1/chat/completions",
			wantHeader: http.Header{
				"Authorization": {"Bearer " + testAPIKey},
				"Content-Type":  {"application/json"},
				"User-Agent":    {DefaultUserAgent},
			},
			wantBody: `{"model":"m","stream":true}`,
		},
		"success: X-Api-Key header": {
			baseURL: "https://llama.example",
			opts:    []ClientOption{WithAPIKey(testAPIKey), WithAPIKeyHeader(APIKeyHeaderXAPIKey), WithUserAgent("agent/2")},
			method:  http.MethodPost,
			path:    "/v1/messages",
			body:    jsontext.Value(`{"max_tokens":8}`),
			wantURL: "https://llama.example/v1/messages",
			wantHeader: http.Header{
				"X-Api-Key":    {testAPIKey},
				"Content-Type": {"application/json"},
				"User-Agent":   {"agent/2"},
			},
			wantBody: `{"max_tokens":8}`,
		},
		"success: queries from the path and from options are merged and sorted": {
			baseURL: "http://localhost:8080",
			method:  http.MethodPost,
			path:    "/slots/0?action=save",
			reqOpts: []RequestOption{WithModelQuery("org/model:Q4"), WithAutoload(false), WithQuery("action", "x"), nil},
			wantURL: "http://localhost:8080/slots/0?action=save&action=x&autoload=false&model=org%2Fmodel%3AQ4",
			wantHeader: http.Header{
				"User-Agent": {DefaultUserAgent},
			},
		},
		"success: custom header and raw body": {
			baseURL: "http://localhost:8080",
			method:  http.MethodPost,
			path:    "/tts",
			body:    &Body{ContentType: "text/plain", Data: []byte("hi")},
			reqOpts: []RequestOption{WithHeader("X-Trace", "1"), WithHeader("User-Agent", "override")},
			wantURL: "http://localhost:8080/tts",
			wantHeader: http.Header{
				"Content-Type": {"text/plain"},
				"User-Agent":   {"override"},
				"X-Trace":      {"1"},
			},
			wantBody: "hi",
		},
		"success: nil *Body sends nothing": {
			baseURL:    "http://localhost:8080",
			method:     http.MethodPost,
			path:       "/props",
			body:       (*Body)(nil),
			wantURL:    "http://localhost:8080/props",
			wantHeader: http.Header{"User-Agent": {DefaultUserAgent}},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, err := New(tt.baseURL, tt.opts...)
			if err != nil {
				t.Fatal(err)
			}
			req, err := c.NewRequest(t.Context(), tt.method, tt.path, tt.body, tt.reqOpts...)
			if err != nil {
				t.Fatalf("NewRequest: %v", err)
			}
			if req.Method != tt.method {
				t.Errorf("Method = %q, want %q", req.Method, tt.method)
			}
			if got := req.URL.String(); got != tt.wantURL {
				t.Errorf("URL = %q, want %q", got, tt.wantURL)
			}
			if diff := gocmp.Diff(tt.wantHeader, req.Header); diff != "" {
				t.Errorf("Header mismatch (-want +got):\n%s", diff)
			}
			var got []byte
			if req.Body != nil {
				got, err = io.ReadAll(req.Body)
				if err != nil {
					t.Fatal(err)
				}
			}
			if string(got) != tt.wantBody {
				t.Errorf("body = %q, want %q", got, tt.wantBody)
			}
			if tt.wantBody != "" {
				if req.GetBody == nil {
					t.Fatalf("GetBody is nil; a retry could not resend the body")
				}
				if req.ContentLength != int64(len(tt.wantBody)) {
					t.Errorf("ContentLength = %d, want %d", req.ContentLength, len(tt.wantBody))
				}
			}
		})
	}
}

func TestNewRequestErrors(t *testing.T) {
	c, err := New("http://localhost:8080")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		method, path string
		body         any
		want         string
	}{
		"error: relative path":    {method: http.MethodGet, path: "health", want: "must start with /"},
		"error: broken query":     {method: http.MethodGet, path: "/x?%zz", want: "invalid query"},
		"error: unencodable body": {method: http.MethodPost, path: "/x", body: map[string]any{"f": func() {}}, want: "encoding request body"},
		"error: invalid method":   {method: "BAD METHOD", path: "/x", want: "building request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := c.NewRequest(t.Context(), tt.method, tt.path, tt.body)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("NewRequest error = %v, want one containing %q", err, tt.want)
			}
		})
	}
}

// TestRequestForAttemptReplaysIdenticalBody checks the body every retry attempt sends, for a JSON body and for a
// multipart/form-data body like the one of /tts.
func TestRequestForAttemptReplaysIdenticalBody(t *testing.T) {
	var form bytes.Buffer
	mw := multipart.NewWriter(&form)
	if err := mw.WriteField("input", "hello"); err != nil {
		t.Fatal(err)
	}
	fw, err := mw.CreateFormFile("speaker_ref", "ref.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.Write(bytes.Repeat([]byte{0, 1, 2, 255}, 4096)); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}

	tests := map[string]struct {
		body any
	}{
		"success: JSON":      {body: map[string]any{"prompt": "a", "n_predict": 4, "stream": false}},
		"success: multipart": {body: &Body{ContentType: mw.FormDataContentType(), Data: form.Bytes()}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, err := New("http://localhost:8080")
			if err != nil {
				t.Fatal(err)
			}
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/x", tt.body)
			if err != nil {
				t.Fatal(err)
			}
			var first []byte
			for n := range 4 {
				r, err := requestForAttempt(req, n+1)
				if err != nil {
					t.Fatalf("attempt %d: %v", n+1, err)
				}
				got, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatal(err)
				}
				if n == 0 {
					if r != req {
						t.Errorf("attempt 1 must send the original request")
					}
					first = got
					continue
				}
				if r == req {
					t.Errorf("attempt %d reused the original request whose body was consumed", n+1)
				}
				if !bytes.Equal(got, first) {
					t.Errorf("attempt %d body differs from attempt 1: %d vs %d bytes", n+1, len(got), len(first))
				}
				if r.Header.Get("Content-Type") != req.Header.Get("Content-Type") {
					t.Errorf("attempt %d lost the Content-Type header", n+1)
				}
			}
			if len(first) == 0 {
				t.Fatalf("attempt 1 sent an empty body")
			}
		})
	}
}

func TestRequestForAttemptWithoutGetBody(t *testing.T) {
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "http://localhost/x", io.NopCloser(strings.NewReader("once")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestForAttempt(req, 2); err == nil {
		t.Errorf("requestForAttempt resent a body it cannot regenerate")
	}
	req, err = http.NewRequestWithContext(t.Context(), http.MethodGet, "http://localhost/x", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestForAttempt(req, 2); err != nil {
		t.Errorf("a request without body must be resendable: %v", err)
	}
}

func TestRetryableTransportError(t *testing.T) {
	refused := &url.Error{Op: "Get", URL: "http://x", Err: &net.OpError{Op: "dial", Err: errors.New("connection refused")}}
	canceledCtx, cancel := context.WithCancel(t.Context())
	cancel()
	tests := map[string]struct {
		ctx    context.Context
		method string
		err    error
		want   bool
	}{
		"retry: GET refused":            {method: http.MethodGet, err: refused, want: true},
		"retry: HEAD refused":           {method: http.MethodHead, err: refused, want: true},
		"retry: OPTIONS refused":        {method: http.MethodOptions, err: refused, want: true},
		"retry: PUT refused":            {method: http.MethodPut, err: refused, want: true},
		"retry: DELETE refused":         {method: http.MethodDelete, err: refused, want: true},
		"no retry: POST refused":        {method: http.MethodPost, err: refused, want: false},
		"no retry: PATCH refused":       {method: http.MethodPatch, err: refused, want: false},
		"no retry: context canceled":    {method: http.MethodGet, err: fmt.Errorf("x: %w", context.Canceled), want: false},
		"no retry: deadline":            {method: http.MethodGet, err: fmt.Errorf("x: %w", context.DeadlineExceeded), want: false},
		"no retry: ctx done, other err": {ctx: canceledCtx, method: http.MethodGet, err: refused, want: false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			ctx := tt.ctx
			if ctx == nil {
				ctx = t.Context()
			}
			if got := retryableTransportError(ctx, tt.method, tt.err); got != tt.want {
				t.Errorf("retryableTransportError(%s, %v) = %t, want %t", tt.method, tt.err, got, tt.want)
			}
		})
	}
}

func TestRetryableStatus(t *testing.T) {
	tests := map[string]struct {
		err  *APIError
		want bool
	}{
		"retry: 503 unavailable":          {err: &APIError{StatusCode: 503, Type: ErrorTypeUnavailable}, want: true},
		"no retry: 503 from a proxy":      {err: &APIError{StatusCode: 503}, want: false},
		"no retry: 500 server error":      {err: &APIError{StatusCode: 500, Type: ErrorTypeServer}, want: false},
		"no retry: 400 invalid request":   {err: &APIError{StatusCode: 400, Type: ErrorTypeInvalidRequest}, want: false},
		"no retry: unavailable with 200":  {err: &APIError{StatusCode: 200, Type: ErrorTypeUnavailable}, want: false},
		"no retry: 401 authentication":    {err: &APIError{StatusCode: 401, Type: ErrorTypeAuthentication}, want: false},
		"no retry: 501 not supported":     {err: &APIError{StatusCode: 501, Type: ErrorTypeNotSupported}, want: false},
		"no retry: 404 not found":         {err: &APIError{StatusCode: 404, Type: ErrorTypeNotFound}, want: false},
		"no retry: context size exceeded": {err: &APIError{StatusCode: 400, Type: ErrorTypeExceedContextSize}, want: false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := retryableStatus(tt.err); got != tt.want {
				t.Errorf("retryableStatus(%+v) = %t, want %t", tt.err, got, tt.want)
			}
		})
	}
}

func TestRetryAfter(t *testing.T) {
	tests := map[string]struct {
		value string
		want  time.Duration
	}{
		"success: seconds":   {value: "3", want: 3 * time.Second},
		"ignored: absent":    {value: "", want: 0},
		"ignored: HTTP date": {value: "Wed, 21 Oct 2015 07:28:00 GMT", want: 0},
		"ignored: negative":  {value: "-1", want: 0},
		"ignored: zero":      {value: "0", want: 0},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			h := http.Header{}
			if tt.value != "" {
				h.Set("Retry-After", tt.value)
			}
			if got := retryAfter(h); got != tt.want {
				t.Errorf("retryAfter(%q) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

// step is one scripted outcome of an attempt.
type step struct {
	val       string
	err       error
	retryable bool
	after     time.Duration
}

func (s step) result() (string, error) {
	if s.err == nil {
		return s.val, nil
	}
	if s.retryable {
		return "", &retryableError{err: s.err, after: s.after}
	}
	return "", s.err
}

// TestRetryLoop drives retry with scripted attempt sequences under a fake clock.
func TestRetryLoop(t *testing.T) {
	errUnavailable := &APIError{StatusCode: 503, Type: ErrorTypeUnavailable, Message: "Loading model"}
	errBad := &APIError{StatusCode: 400, Type: ErrorTypeInvalidRequest}
	policy := RetryPolicy{
		MaxAttempts:         4,
		InitialInterval:     100 * time.Millisecond,
		MaxInterval:         time.Second,
		Multiplier:          2,
		RandomizationFactor: 0.5,
		MaxElapsedTime:      time.Minute,
	}
	tests := map[string]struct {
		policy       RetryPolicy
		steps        []step
		wantAttempts int
		wantVal      string
		wantErr      error
		wantGapMin   time.Duration
		wantGapMax   time.Duration
	}{
		"success: first attempt": {
			policy:       policy,
			steps:        []step{{val: "ok"}},
			wantAttempts: 1,
			wantVal:      "ok",
		},
		"success: failure then success": {
			policy:       policy,
			steps:        []step{{err: errUnavailable, retryable: true}, {err: errUnavailable, retryable: true}, {val: "ok"}},
			wantAttempts: 3,
			wantVal:      "ok",
			wantGapMin:   50 * time.Millisecond,
			wantGapMax:   300 * time.Millisecond,
		},
		"error: retries exhausted returns the last error unmarked": {
			policy: policy,
			steps: []step{
				{err: errors.New("e1"), retryable: true},
				{err: errors.New("e2"), retryable: true},
				{err: errors.New("e3"), retryable: true},
				{err: errUnavailable, retryable: true},
				{val: "never"},
			},
			wantAttempts: 4,
			wantErr:      errUnavailable,
		},
		"error: permanent error stops at once": {
			policy:       policy,
			steps:        []step{{err: errBad}, {val: "never"}},
			wantAttempts: 1,
			wantErr:      errBad,
		},
		"error: permanent after retryable": {
			policy:       policy,
			steps:        []step{{err: errUnavailable, retryable: true}, {err: errBad}, {val: "never"}},
			wantAttempts: 2,
			wantErr:      errBad,
		},
		"success: retries disabled by MaxAttempts 1": {
			policy:       RetryPolicy{MaxAttempts: 1},
			steps:        []step{{err: errUnavailable, retryable: true}, {val: "never"}},
			wantAttempts: 1,
			wantErr:      errUnavailable,
		},
		"success: zero policy sends once": {
			policy:       RetryPolicy{},
			steps:        []step{{err: errUnavailable, retryable: true}, {val: "never"}},
			wantAttempts: 1,
			wantErr:      errUnavailable,
		},
		"success: Retry-After replaces the backoff wait": {
			policy:       policy,
			steps:        []step{{err: errUnavailable, retryable: true, after: 700 * time.Millisecond}, {val: "ok"}},
			wantAttempts: 2,
			wantVal:      "ok",
			wantGapMin:   700 * time.Millisecond,
			wantGapMax:   700 * time.Millisecond,
		},
		"success: Retry-After is capped by MaxInterval": {
			policy:       policy,
			steps:        []step{{err: errUnavailable, retryable: true, after: time.Hour}, {val: "ok"}},
			wantAttempts: 2,
			wantVal:      "ok",
			wantGapMin:   time.Second,
			wantGapMax:   time.Second,
		},
		"error: MaxElapsedTime stops before the wait would pass it": {
			policy: RetryPolicy{
				MaxAttempts: 100, InitialInterval: time.Second, MaxInterval: time.Second, Multiplier: 1,
				MaxElapsedTime: 2500 * time.Millisecond,
			},
			steps: []step{
				{err: errors.New("e1"), retryable: true},
				{err: errors.New("e2"), retryable: true},
				{err: errUnavailable, retryable: true},
				{val: "never"},
			},
			wantAttempts: 3,
			wantErr:      errUnavailable,
			wantGapMin:   time.Second,
			wantGapMax:   time.Second,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var starts []time.Time
				got, err := retry(t.Context(), tt.policy, func(_ context.Context, n int) (string, error) {
					if n != len(starts)+1 {
						t.Errorf("attempt number = %d, want %d", n, len(starts)+1)
					}
					starts = append(starts, time.Now())
					if n > len(tt.steps) {
						t.Fatalf("attempt %d beyond the script", n)
					}
					return tt.steps[n-1].result()
				})
				if len(starts) != tt.wantAttempts {
					t.Errorf("attempts = %d, want %d", len(starts), tt.wantAttempts)
				}
				if got != tt.wantVal {
					t.Errorf("value = %q, want %q", got, tt.wantVal)
				}
				if tt.wantErr == nil && err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if tt.wantErr != nil && (!errors.Is(err, tt.wantErr) || errors.Unwrap(err) != nil) {
					t.Errorf("error = %#v, want exactly %#v (no retry mark, no wrapping)", err, tt.wantErr)
				}
				if _, ok := errors.AsType[*retryableError](err); ok {
					t.Errorf("returned error still carries the retry mark: %v", err)
				}
				for i := 1; i < len(starts); i++ {
					gap := starts[i].Sub(starts[i-1])
					if tt.wantGapMax > 0 && (gap < tt.wantGapMin || gap > tt.wantGapMax) {
						t.Errorf("wait before attempt %d = %v, want within [%v, %v]", i+1, gap, tt.wantGapMin, tt.wantGapMax)
					}
				}
			})
		})
	}
}

// TestRetryWaitsAreBounded checks that no wait exceeds MaxInterval, including an InitialInterval above it and
// the upward jitter of a capped interval.
func TestRetryWaitsAreBounded(t *testing.T) {
	tests := map[string]RetryPolicy{
		"success: growth reaches the cap": {
			MaxAttempts: 12, InitialInterval: 100 * time.Millisecond, MaxInterval: 400 * time.Millisecond, Multiplier: 3, RandomizationFactor: 0.5,
		},
		"success: initial interval above the cap": {
			MaxAttempts: 6, InitialInterval: 2 * time.Second, MaxInterval: time.Second, Multiplier: 2, RandomizationFactor: 0.5,
		},
		"success: full jitter at the cap": {
			MaxAttempts: 20, InitialInterval: time.Second, MaxInterval: time.Second, Multiplier: 1, RandomizationFactor: 1,
		},
	}
	for name, p := range tests {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var starts []time.Time
				_, err := retry(t.Context(), p, func(context.Context, int) (int, error) {
					starts = append(starts, time.Now())
					return 0, &retryableError{err: errors.New("down")}
				})
				if err == nil || err.Error() != "down" {
					t.Fatalf("error = %v, want the last attempt error", err)
				}
				if len(starts) != p.MaxAttempts {
					t.Fatalf("attempts = %d, want %d", len(starts), p.MaxAttempts)
				}
				for i := 1; i < len(starts); i++ {
					if gap := starts[i].Sub(starts[i-1]); gap < 0 || gap > p.MaxInterval {
						t.Errorf("wait before attempt %d = %v, want within [0, %v]", i+1, gap, p.MaxInterval)
					}
				}
			})
		})
	}
}

func TestRetryCancellation(t *testing.T) {
	t.Run("error: canceled while waiting keeps both errors", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			lastErr := &APIError{StatusCode: 503, Type: ErrorTypeUnavailable}
			attempts := 0
			go func() {
				time.Sleep(50 * time.Millisecond)
				cancel()
			}()
			_, err := retry(ctx, RetryPolicy{MaxAttempts: 10, InitialInterval: time.Second, MaxInterval: time.Second, Multiplier: 1},
				func(context.Context, int) (int, error) {
					attempts++
					return 0, &retryableError{err: lastErr}
				})
			if attempts != 1 {
				t.Errorf("attempts = %d, want 1: the cancel happened during the first wait", attempts)
			}
			if !errors.Is(err, context.Canceled) {
				t.Errorf("error %v does not match context.Canceled", err)
			}
			if apiErr, ok := errors.AsType[*APIError](err); !ok || apiErr != lastErr {
				t.Errorf("error %v does not carry the last attempt's *APIError", err)
			}
		})
	})
	t.Run("error: canceled by the attempt itself", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			attempts := 0
			_, err := retry(ctx, DefaultRetryPolicy(), func(context.Context, int) (int, error) {
				attempts++
				cancel()
				return 0, &retryableError{err: errors.New("refused")}
			})
			if attempts != 1 || !errors.Is(err, context.Canceled) {
				t.Errorf("attempts = %d, err = %v; want 1 attempt and context.Canceled", attempts, err)
			}
		})
	})
	t.Run("error: transport error from a canceled request is not retried", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			attempts := 0
			_, err := retry(ctx, DefaultRetryPolicy(), func(ctx context.Context, _ int) (int, error) {
				attempts++
				cancel()
				doErr := &url.Error{Op: "Get", URL: "http://x", Err: context.Canceled}
				if retryableTransportError(ctx, http.MethodGet, doErr) {
					return 0, &retryableError{err: doErr}
				}
				return 0, doErr
			})
			if attempts != 1 || !errors.Is(err, context.Canceled) {
				t.Errorf("attempts = %d, err = %v; want 1 attempt and context.Canceled", attempts, err)
			}
		})
	})
}

// TestCallConnectionFailure uses a closed local port: the dial fails before any response, which is what the
// client does for a server that is not listening.
func TestCallConnectionFailure(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	c, err := New("http://"+addr, WithAPIKey(testAPIKey), WithLogger(logger),
		WithRetry(RetryPolicy{MaxAttempts: 3, InitialInterval: time.Millisecond, MaxInterval: time.Millisecond, Multiplier: 1}))
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()

	tests := map[string]struct {
		method       string
		path         string
		opts         []RequestOption
		wantAttempts string
	}{
		"retry: GET is retried up to the limit": {method: http.MethodGet, path: "/props", wantAttempts: "attempts=3"},
		"no retry: POST is sent once":           {method: http.MethodPost, path: "/props", wantAttempts: "attempts=1"},
		"no retry: WithNoRetry":                 {method: http.MethodGet, path: "/props", opts: []RequestOption{WithNoRetry()}, wantAttempts: "attempts=1"},
		"retry: busy slot option":               {method: http.MethodGet, path: "/slots", opts: []RequestOption{WithFailOnNoSlot()}, wantAttempts: "attempts=3"},
		"retry: busy slot query option":         {method: http.MethodGet, path: "/slots", opts: []RequestOption{WithQuery("fail_on_no_slot", "1")}, wantAttempts: "attempts=3"},
		"retry: busy slot query in path":        {method: http.MethodGet, path: "/slots?fail_on_no_slot=1", wantAttempts: "attempts=3"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			logs.Reset()
			_, err := Call[map[string]jsontext.Value](t.Context(), c, tt.method, tt.path, map[string]int{"n": 1}, tt.opts...)
			if err == nil {
				t.Fatalf("Call to a closed port succeeded")
			}
			if _, ok := errors.AsType[*APIError](err); ok {
				t.Errorf("a connection failure must not be an *APIError: %v", err)
			}
			for _, out := range []string{err.Error(), fmt.Sprintf("%+v", err), fmt.Sprintf("%#v", err), logs.String()} {
				if strings.Contains(out, testAPIKey) {
					t.Errorf("API key leaked: %s", out)
				}
			}
			if !strings.Contains(logs.String(), tt.wantAttempts) {
				t.Errorf("log does not record %s:\n%s", tt.wantAttempts, logs.String())
			}
			path, _, _ := strings.Cut(tt.path, "?")
			if !strings.Contains(logs.String(), "path="+path) || !strings.Contains(logs.String(), "method="+tt.method) {
				t.Errorf("log lacks method or path:\n%s", logs.String())
			}
		})
	}
}
