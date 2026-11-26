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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"strings"
	"unicode/utf8"
)

// ErrorType is the "type" member of an error returned by llama-server.
type ErrorType string

// The error types llama-server reports, with the HTTP status code each one is sent with.
const (
	// ErrorTypeInvalidRequest is sent with status 400.
	ErrorTypeInvalidRequest ErrorType = "invalid_request_error"
	// ErrorTypeAuthentication is sent with status 401.
	ErrorTypeAuthentication ErrorType = "authentication_error"
	// ErrorTypeNotFound is sent with status 404.
	ErrorTypeNotFound ErrorType = "not_found_error"
	// ErrorTypeServer is sent with status 500.
	ErrorTypeServer ErrorType = "server_error"
	// ErrorTypePermission is sent with status 403.
	ErrorTypePermission ErrorType = "permission_error"
	// ErrorTypeNotSupported is sent with status 501.
	ErrorTypeNotSupported ErrorType = "not_supported_error"
	// ErrorTypeUnavailable is sent with status 503, for example while the model is loading.
	ErrorTypeUnavailable ErrorType = "unavailable_error"
	// ErrorTypeExceedContextSize is sent with status 400 when the prompt does not fit the slot context.
	ErrorTypeExceedContextSize ErrorType = "exceed_context_size_error"
)

// maxErrorMessage bounds how much of a non-JSON error body is copied into APIError.Message.
const maxErrorMessage = 512

// maxErrorBody bounds how much of an error response body is read.
const maxErrorBody = 1 << 20

// APIError is an error reported by llama-server, either as an HTTP error response or as an error event inside a
// stream.
//
// A body that is not the server's JSON error object (for example an HTML page from a proxy) still produces an
// APIError: StatusCode is set, Type is empty and Message holds the start of the body.
type APIError struct {
	// StatusCode is the HTTP status code of the response. For an error delivered inside a stream it is the status
	// of the stream response (200), and Code carries the status the server assigned to the error.
	StatusCode int
	// Code is the status assigned to the JSON error, which can differ from StatusCode and is zero if omitted.
	Code int
	// Message is the server's explanation, or a short body snippet or HTTP status text for a non-JSON error.
	Message string
	// Type is the server's machine-readable error category, or empty when none was supplied.
	Type ErrorType
	// NPromptTokens is the prompt length in tokens, set for ErrorTypeExceedContextSize.
	NPromptTokens int
	// NCtx is the slot capacity in tokens, set for ErrorTypeExceedContextSize.
	NCtx int
	// Raw holds the error response body or in-stream JSON payload, truncated to 1 MiB.
	Raw []byte
	// Header is the header of the response that carried the error.
	Header http.Header
}

// Error implements the error interface.
func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString("llama: ")
	if e.StatusCode != 0 {
		fmt.Fprintf(&b, "HTTP %d", e.StatusCode)
	}
	if e.Type != "" {
		fmt.Fprintf(&b, " %s", e.Type)
	}
	if e.Code != 0 && e.Code != e.StatusCode {
		fmt.Fprintf(&b, " (code %d)", e.Code)
	}
	if e.Message != "" {
		fmt.Fprintf(&b, ": %s", e.Message)
	}
	if e.Type == ErrorTypeExceedContextSize {
		fmt.Fprintf(&b, " (n_prompt_tokens=%d, n_ctx=%d)", e.NPromptTokens, e.NCtx)
	}
	return b.String()
}

// errorObject is the JSON error object produced by the server's format_error_response.
type errorObject struct {
	Code          int                       `json:"code,omitzero"`
	Message       string                    `json:"message,omitzero"`
	Type          ErrorType                 `json:"type,omitzero"`
	NPromptTokens int                       `json:"n_prompt_tokens,omitzero"`
	NCtx          int                       `json:"n_ctx,omitzero"`
	Extra         map[string]jsontext.Value `json:",embed"`
}

// errorEnvelope is the {"error": {...}} wrapper used by HTTP error responses and by in-stream errors of the
// "data:"-only stream dialects.
type errorEnvelope struct {
	Error *errorObject              `json:"error"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// boundedErrorBody keeps the raw prefix without retaining a larger event buffer.
func boundedErrorBody(raw []byte) []byte {
	if len(raw) > maxErrorBody {
		return bytes.Clone(raw[:maxErrorBody])
	}
	return raw
}

func newAPIError(status int, obj *errorObject, raw []byte, header http.Header) *APIError {
	return &APIError{
		StatusCode:    status,
		Code:          obj.Code,
		Message:       obj.Message,
		Type:          obj.Type,
		NPromptTokens: obj.NPromptTokens,
		NCtx:          obj.NCtx,
		Raw:           boundedErrorBody(raw),
		Header:        header,
	}
}

// decodeAPIError converts an HTTP error response body into an APIError. It never fails: a body that is not the
// server's error object is reported through Message.
func decodeAPIError(status int, header http.Header, body []byte) *APIError {
	var env errorEnvelope
	if err := json.Unmarshal(body, &env, jsonOptions); err == nil && env.Error != nil {
		return newAPIError(status, env.Error, body, header)
	}
	// Some proxies and older servers send the error object without the envelope.
	var obj errorObject
	if err := json.Unmarshal(body, &obj, jsonOptions); err == nil && (obj.Type != "" || obj.Message != "") {
		return newAPIError(status, &obj, body, header)
	}
	return &APIError{
		StatusCode: status,
		Message:    bodySnippet(body, status),
		Raw:        body,
		Header:     header,
	}
}

// ReadAPIError reads and closes the body of an error response and converts it into an *APIError. It serves code
// that sends requests itself, such as a WebSocket handshake that the server refused. A failure to read the body
// still yields an APIError built from the status and whatever was read.
func ReadAPIError(resp *http.Response) *APIError {
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	return decodeAPIError(resp.StatusCode, resp.Header, body)
}

// bodySnippet returns the start of a non-JSON body as valid UTF-8, or the status text when the body is empty.
func bodySnippet(body []byte, status int) string {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return http.StatusText(status)
	}
	if len(body) > maxErrorMessage {
		// Cut on a rune boundary so a multi-byte character is not reported as invalid UTF-8.
		cut := maxErrorMessage
		for cut > 0 && !utf8.RuneStart(body[cut]) {
			cut--
		}
		body = body[:cut]
	}
	return strings.ToValidUTF8(string(body), "�")
}
