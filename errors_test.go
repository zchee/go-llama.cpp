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
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// The bodies below follow format_error_response in tools/server/server-common.cpp and server_task_result_error in
// tools/server/server-task.cpp of llama.cpp v1.1.0.
func TestDecodeAPIError(t *testing.T) {
	tests := map[string]struct {
		status int
		body   string
		want   *APIError
	}{
		"success: invalid_request_error": {
			status: 400,
			body:   `{"error":{"code":400,"message":"\"prompt\" is required","type":"invalid_request_error"}}`,
			want:   &APIError{StatusCode: 400, Code: 400, Message: `"prompt" is required`, Type: ErrorTypeInvalidRequest},
		},
		"success: authentication_error": {
			status: 401,
			body:   `{"error":{"message":"Invalid API Key","type":"authentication_error","code":401}}`,
			want:   &APIError{StatusCode: 401, Code: 401, Message: "Invalid API Key", Type: ErrorTypeAuthentication},
		},
		"success: not_found_error": {
			status: 404,
			body:   `{"error":{"message":"File Not Found","type":"not_found_error","code":404}}`,
			want:   &APIError{StatusCode: 404, Code: 404, Message: "File Not Found", Type: ErrorTypeNotFound},
		},
		"success: server_error": {
			status: 500,
			body:   `{"error":{"code":500,"message":"unknown error","type":"server_error"}}`,
			want:   &APIError{StatusCode: 500, Code: 500, Message: "unknown error", Type: ErrorTypeServer},
		},
		"success: permission_error": {
			status: 403,
			body:   `{"error":{"code":403,"message":"forbidden","type":"permission_error"}}`,
			want:   &APIError{StatusCode: 403, Code: 403, Message: "forbidden", Type: ErrorTypePermission},
		},
		"success: not_supported_error": {
			status: 501,
			body:   `{"error":{"code":501,"message":"This server does not support embeddings.","type":"not_supported_error"}}`,
			want:   &APIError{StatusCode: 501, Code: 501, Message: "This server does not support embeddings.", Type: ErrorTypeNotSupported},
		},
		"success: unavailable_error": {
			status: 503,
			body:   `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`,
			want:   &APIError{StatusCode: 503, Code: 503, Message: "Loading model", Type: ErrorTypeUnavailable},
		},
		"success: exceed_context_size_error with sizes": {
			status: 400,
			body:   `{"error":{"code":400,"message":"request exceeds the available context size","type":"exceed_context_size_error","n_prompt_tokens":1024,"n_ctx":512}}`,
			want: &APIError{
				StatusCode: 400, Code: 400, Message: "request exceeds the available context size",
				Type: ErrorTypeExceedContextSize, NPromptTokens: 1024, NCtx: 512,
			},
		},
		"success: bare error object without envelope": {
			status: 400,
			body:   `{"code":400,"message":"bad","type":"invalid_request_error"}`,
			want:   &APIError{StatusCode: 400, Code: 400, Message: "bad", Type: ErrorTypeInvalidRequest},
		},
		"fallback: HTML from a proxy": {
			status: 502,
			body:   "<html><body>Bad Gateway</body></html>\n",
			want:   &APIError{StatusCode: 502, Message: "<html><body>Bad Gateway</body></html>"},
		},
		"fallback: empty body": {
			status: 500,
			body:   "",
			want:   &APIError{StatusCode: 500, Message: "Internal Server Error"},
		},
		"fallback: JSON without error members": {
			status: 500,
			body:   `{"status":"broken"}`,
			want:   &APIError{StatusCode: 500, Message: `{"status":"broken"}`},
		},
		"fallback: plain text from ex_wrapper double failure": {
			status: 500,
			body:   "Internal Server Error",
			want:   &APIError{StatusCode: 500, Message: "Internal Server Error"},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			header := http.Header{"Content-Type": {"application/json"}}
			got := decodeAPIError(tt.status, header, []byte(tt.body))
			if diff := gocmp.Diff(tt.want, got, cmpopts.IgnoreFields(APIError{}, "Raw", "Header")); diff != "" {
				t.Errorf("decodeAPIError(%d, %q) mismatch (-want +got):\n%s", tt.status, tt.body, diff)
			}
			if string(got.Raw) != tt.body {
				t.Errorf("Raw = %q, want the body %q", got.Raw, tt.body)
			}
			if got.Header.Get("Content-Type") != "application/json" {
				t.Errorf("Header was not kept: %v", got.Header)
			}
			var err error = got
			if apiErr, ok := errors.AsType[*APIError](fmt.Errorf("wrapped: %w", err)); !ok || apiErr != got {
				t.Errorf("errors.AsType did not find the *APIError through a wrap")
			}
		})
	}
}

func TestBodySnippetLimits(t *testing.T) {
	long := strings.Repeat("あ", maxErrorMessage) // 3 bytes per rune, so the limit cuts inside a rune
	got := bodySnippet([]byte(long), 500)
	if len(got) > maxErrorMessage {
		t.Errorf("len(snippet) = %d, want at most %d", len(got), maxErrorMessage)
	}
	if strings.ContainsRune(got, '�') {
		t.Errorf("snippet contains a replacement character; the cut must fall on a rune boundary: %q", got[len(got)-10:])
	}
	if !strings.HasPrefix(long, got) {
		t.Errorf("snippet is not a prefix of the body")
	}
}

func TestReadAPIErrorClosesBody(t *testing.T) {
	body := &closeRecorder{Reader: strings.NewReader(`{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`)}
	resp := &http.Response{StatusCode: 503, Header: http.Header{}, Body: body}
	got := ReadAPIError(resp)
	if got.Type != ErrorTypeUnavailable {
		t.Errorf("Type = %q, want %q", got.Type, ErrorTypeUnavailable)
	}
	if !body.closed {
		t.Errorf("ReadAPIError did not close the response body")
	}
}

func TestAPIErrorMessage(t *testing.T) {
	tests := map[string]struct {
		err  *APIError
		want string
	}{
		"success: typed error": {
			err:  &APIError{StatusCode: 401, Code: 401, Message: "Invalid API Key", Type: ErrorTypeAuthentication},
			want: "llama: HTTP 401 authentication_error: Invalid API Key",
		},
		"success: context size": {
			err:  &APIError{StatusCode: 400, Code: 400, Message: "too long", Type: ErrorTypeExceedContextSize, NPromptTokens: 9, NCtx: 8},
			want: "llama: HTTP 400 exceed_context_size_error: too long (n_prompt_tokens=9, n_ctx=8)",
		},
		"success: in-stream error keeps the error code": {
			err:  &APIError{StatusCode: 200, Code: 500, Message: "boom", Type: ErrorTypeServer},
			want: "llama: HTTP 200 server_error (code 500): boom",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}
		})
	}
}

// closeRecorder is a response body that records Close.
type closeRecorder struct {
	io.Reader
	closed bool
}

func (c *closeRecorder) Close() error {
	c.closed = true
	return nil
}
