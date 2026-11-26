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
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"testing"
	"time"

	gocmp "github.com/google/go-cmp/cmp"
)

// errTransportDown is the sentinel the dumping transport wraps, to check that redaction keeps errors.Is working.
var errTransportDown = errors.New("transport down")

// dumpError is an error that quotes the whole request, headers included, as some transports and proxies do.
type dumpError struct {
	dump string
}

func (e *dumpError) Error() string { return "request failed: " + e.dump }

func (e *dumpError) Unwrap() error { return errTransportDown }

// dumpingTransport fails every request with a dumpError, and also quotes the credential headers with %q.
func dumpingTransport() http.RoundTripper {
	return roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		dump, err := httputil.DumpRequest(r, false)
		if err != nil {
			return nil, err
		}
		quoted := fmt.Sprintf(" quoted=%q %q", r.Header.Get("Authorization"), r.Header.Get("X-Api-Key"))
		return nil, &dumpError{dump: string(dump) + quoted}
	})
}

// specialKey contains characters that URL escaping and Go quoting spell differently.
const specialKey = `sk-a/b+c d"e\f`

func TestErrorsNeverCarryAPIKey(t *testing.T) {
	keyForms := func(key string) []string {
		q := fmt.Sprintf("%q", key)
		return []string{key, url.QueryEscape(key), url.PathEscape(key), q[1 : len(q)-1]}
	}
	tests := map[string]struct {
		key    string
		header APIKeyHeader
	}{
		"success: bearer":    {key: testAPIKey, header: APIKeyHeaderAuthorization},
		"success: X-Api-Key": {key: testAPIKey, header: APIKeyHeaderXAPIKey},
		"success: escaped forms of a key with special characters": {key: specialKey, header: APIKeyHeaderXAPIKey},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			c, err := New("http://127.0.0.1:8080",
				WithAPIKey(tt.key), WithAPIKeyHeader(tt.header),
				WithHTTPClient(&http.Client{Transport: dumpingTransport()}),
				WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))),
				WithRetry(RetryPolicy{MaxAttempts: 2, InitialInterval: time.Millisecond, MaxInterval: time.Millisecond, Multiplier: 1}))
			if err != nil {
				t.Fatal(err)
			}
			calls := map[string]func() error{
				"Call": func() error {
					_, err := Call[HealthResponse](t.Context(), c, http.MethodGet, "/props", nil)
					return err
				},
				"CallRaw": func() error {
					resp, err := CallRaw(t.Context(), c, http.MethodPost, "/tts", map[string]string{"input": "x"})
					if resp != nil {
						resp.Body.Close()
					}
					return err
				},
				"Stream": func() error {
					for _, err := range Stream[HealthResponse](t.Context(), c, http.MethodPost, "/completion", nil) {
						return err
					}
					return nil
				},
				"Health": func() error {
					_, err := c.Health(t.Context())
					return err
				},
			}
			for method, call := range calls {
				logs.Reset()
				err := call()
				if err == nil {
					t.Fatalf("%s: the failing transport produced no error", method)
				}
				if !errors.Is(err, errTransportDown) {
					t.Errorf("%s: errors.Is lost the transport's sentinel: %v", method, err)
				}
				inner, ok := errors.AsType[*dumpError](err)
				if !ok {
					t.Fatalf("%s: errors.As cannot reach the transport's error type: %v", method, err)
				}
				if !strings.Contains(inner.dump, tt.key) {
					t.Fatalf("%s: the transport error does not quote the key, so the test proves nothing", method)
				}
				outputs := []string{
					err.Error(), fmt.Sprintf("%v", err), fmt.Sprintf("%+v", err), fmt.Sprintf("%#v", err),
					fmt.Sprintf("%s", err), fmt.Sprintf("%q", err), logs.String(),
				}
				for _, out := range outputs {
					for _, form := range keyForms(tt.key) {
						if strings.Contains(out, form) {
							t.Errorf("%s: output carries the key as %q:\n%s", method, form, out)
						}
					}
				}
				if !strings.Contains(err.Error(), redactedText) {
					t.Errorf("%s: error text lacks the redaction marker: %s", method, err)
				}
			}
		})
	}
}

func TestRedactDiagnosticForms(t *testing.T) {
	const key = "fake/a+b"
	const quotedKey = `fake/"b\c`
	tests := map[string]struct {
		key  string
		err  error
		want string
	}{
		"success: key only in response header":           {key, &APIError{StatusCode: http.StatusForbidden, Header: http.Header{"X-Api-Key": {key}}}, "llama: HTTP 403"},
		"success: wrapped transport request dump":        {key, fmt.Errorf("sending: %w", &dumpError{dump: "X-Api-Key: " + key}), "sending: request failed: X-Api-Key: [REDACTED]"},
		"success: lower-case percent escape":             {key, &url.Error{Op: "Get", URL: "https://example.invalid/fake%2fa+b", Err: errTransportDown}, `Get "https://example.invalid/[REDACTED]": transport down`},
		"success: escaped and literal bytes":             {key, errors.New("%66ak%65/a%2Bb"), redactedText},
		"success: literal letters remain case-sensitive": {key, errors.New("Fake%2fa+b"), "Fake%2fa+b"},
		"success: quoted canonical headers":              {quotedKey, fmt.Errorf("%#v", http.Header{"X-Api-Key": {quotedKey}, "Authorization": {"Bearer " + quotedKey}}), `http.Header{"Authorization":[]string{"Bearer [REDACTED]"}, "X-Api-Key":[]string{"[REDACTED]"}}`},
		"success: HTML-escaped JSON string":              {`fake<&"`, errors.New(`{"key":"fake<&\""}`), `{"key":"[REDACTED]"}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got := newSecret(tt.key).redactError(tt.err)
			if !errors.Is(got, tt.err) || got.Error() != tt.want {
				t.Error("redaction changed the error chain or did not produce the expected diagnostic")
			}
			for _, verb := range []string{"%v", "%+v", "%#v", "%s", "%q"} {
				want := tt.want
				if verb == "%q" {
					want = fmt.Sprintf("%q", want)
				}
				if out := fmt.Sprintf(verb, got); out != want {
					t.Errorf("format %s: got %s, want %s", verb, out, want)
				}
			}
			if orig, ok := errors.AsType[*APIError](tt.err); ok {
				if inner, ok := errors.AsType[*APIError](got); !ok || inner != orig {
					t.Error("lost original APIError")
				}
			}
		})
	}
}

func TestRedactErrorPreservesNilPointer(t *testing.T) {
	c, err := New("http://localhost", WithAPIKey(testAPIKey))
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct{ err error }{
		"success: API error":    {(*APIError)(nil)},
		"success: custom error": {(*dumpError)(nil)},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := c.RedactError(tt.err); !gocmp.Equal(tt.err, got) {
				t.Errorf("RedactError returned %T, want unchanged %T", got, tt.err)
			}
		})
	}
}

func TestRedactErrorKeepsOtherErrors(t *testing.T) {
	c, err := New("http://127.0.0.1:8080", WithAPIKey(testAPIKey))
	if err != nil {
		t.Fatal(err)
	}
	plain := errors.New("connection refused")
	if got := c.RedactError(plain); !errors.Is(got, plain) || got.Error() != plain.Error() {
		t.Errorf("RedactError did not preserve the cause and message: %v", got)
	}
	if got := c.RedactError(nil); got != nil {
		t.Errorf("RedactError(nil) = %v", got)
	}
	noKey, err := New("http://127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
	withText := errors.New("contains sk anything")
	if got := noKey.RedactError(withText); !errors.Is(got, withText) || errors.Unwrap(got) != nil {
		t.Errorf("a client without a key changed an error: %v", got)
	}
}

// TestClientValueNeverPrintsAPIKey prints copies of a Client, which bypass methods with pointer receivers.
func TestClientValueNeverPrintsAPIKey(t *testing.T) {
	c, err := New("http://127.0.0.1:8080", WithAPIKey(testAPIKey))
	if err != nil {
		t.Fatal(err)
	}
	type holder struct {
		C     Client
		inner Client
	}
	var buf bytes.Buffer
	slog.New(slog.NewJSONHandler(&buf, nil)).Info("copy", "c", *c, "h", holder{C: *c, inner: *c})
	outputs := []string{
		fmt.Sprintf("%#v", *c), fmt.Sprintf("%+v", *c), fmt.Sprintf("%v", *c), fmt.Sprintf("%s", *c),
		fmt.Sprintf("%#v", holder{C: *c, inner: *c}), fmt.Sprintf("%+v", []Client{*c}),
		fmt.Sprintf("%#v", c), buf.String(),
	}
	for _, out := range outputs {
		if strings.Contains(out, testAPIKey) {
			t.Errorf("output carries the API key: %s", out)
		}
	}
}

// TestRedirectKeepsCredentialsOnOrigin calls the redirect policy of the client's HTTP client with the request a
// redirect would send.
func TestRedirectKeepsCredentialsOnOrigin(t *testing.T) {
	tests := map[string]struct {
		target   string
		wantKept bool
	}{
		"kept: same scheme and host": {target: "http://llama.example:8080/other", wantKept: true},
		"kept: host case differs":    {target: "http://LLAMA.example:8080/other", wantKept: true},
		"dropped: other host":        {target: "http://evil.example:8080/x", wantKept: false},
		"dropped: subdomain":         {target: "http://sub.llama.example:8080/x", wantKept: false},
		"dropped: other port":        {target: "http://llama.example:9090/x", wantKept: false},
		"dropped: scheme change":     {target: "https://llama.example:8080/x", wantKept: false},
	}
	for _, header := range []APIKeyHeader{APIKeyHeaderAuthorization, APIKeyHeaderXAPIKey} {
		for name, tt := range tests {
			t.Run(fmt.Sprintf("%s/header %d", name, header), func(t *testing.T) {
				userCalls := 0
				user := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
					userCalls++
					return nil
				}}
				userPolicy := user.CheckRedirect
				c, err := New("http://llama.example:8080", WithAPIKey(testAPIKey), WithAPIKeyHeader(header), WithHTTPClient(user))
				if err != nil {
					t.Fatal(err)
				}
				orig, err := c.NewRequest(t.Context(), http.MethodGet, "/v1/models", nil)
				if err != nil {
					t.Fatal(err)
				}
				next := orig.Clone(t.Context())
				if next.URL, err = url.Parse(tt.target); err != nil {
					t.Fatal(err)
				}
				if err := c.HTTPClient().CheckRedirect(next, []*http.Request{orig}); err != nil {
					t.Fatalf("CheckRedirect: %v", err)
				}
				kept := next.Header.Get("Authorization") != "" || next.Header.Get("X-Api-Key") != ""
				if kept != tt.wantKept {
					t.Errorf("credential kept = %t, want %t (headers %v)", kept, tt.wantKept, next.Header)
				}
				if userCalls != 1 {
					t.Errorf("the caller's redirect policy ran %d times, want 1", userCalls)
				}
				if c.HTTPClient() == user || fmt.Sprintf("%p", user.CheckRedirect) != fmt.Sprintf("%p", userPolicy) {
					t.Errorf("the caller's client was modified")
				}
				if orig.Header.Get("Authorization") == "" && orig.Header.Get("X-Api-Key") == "" {
					t.Errorf("the original request lost its credential")
				}
			})
		}
	}
}

func TestRedirectPolicyLimits(t *testing.T) {
	refuse := errors.New("no redirects")
	tests := map[string]struct {
		user    func(*http.Request, []*http.Request) error
		via     int
		wantErr error
		wantMsg string
	}{
		"success: default policy below the limit":  {via: 9},
		"error: default policy at the limit":       {via: 10, wantMsg: "stopped after 10 redirects"},
		"error: the caller's policy refuses":       {user: func(*http.Request, []*http.Request) error { return refuse }, via: 1, wantErr: refuse},
		"success: the caller's policy allows more": {user: func(*http.Request, []*http.Request) error { return nil }, via: 50},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, err := New("http://llama.example", WithHTTPClient(&http.Client{CheckRedirect: tt.user}))
			if err != nil {
				t.Fatal(err)
			}
			req, err := c.NewRequest(t.Context(), http.MethodGet, "/x", nil)
			if err != nil {
				t.Fatal(err)
			}
			via := make([]*http.Request, tt.via)
			for i := range via {
				via[i] = req
			}
			err = c.HTTPClient().CheckRedirect(req.Clone(t.Context()), via)
			switch {
			case tt.wantErr != nil:
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("CheckRedirect = %v, want %v", err, tt.wantErr)
				}
			case tt.wantMsg != "":
				if err == nil || err.Error() != tt.wantMsg {
					t.Errorf("CheckRedirect = %v, want %q", err, tt.wantMsg)
				}
			default:
				if err != nil {
					t.Errorf("CheckRedirect = %v, want nil", err)
				}
			}
		})
	}
}
