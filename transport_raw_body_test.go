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
	"io"
	"net/http"
	"strings"
	"testing"
)

type writeToErrorBody struct {
	io.ReadCloser
	err error
}

func (b writeToErrorBody) WriteTo(w io.Writer) (int64, error) {
	n, err := io.Copy(w, b.ReadCloser)
	if err != nil {
		return n, err
	}
	return n, b.err
}

func TestCallRawBodyWriterTo(t *testing.T) {
	const payload = "audio"
	writeErr := errors.New("writing body for " + testAPIKey)
	tests := map[string]struct {
		body       io.ReadCloser
		wantErr    error
		wantText   string
		wantWriter bool
	}{
		"success: body without WriterTo keeps Read and Close only": {
			body: io.NopCloser(io.LimitReader(strings.NewReader(payload), int64(len(payload)))),
		},
		"error: body with WriterTo keeps the copy path and redacts its error": {
			body:       writeToErrorBody{ReadCloser: io.NopCloser(strings.NewReader(payload)), err: writeErr},
			wantErr:    writeErr,
			wantText:   "writing body for " + redactedText,
			wantWriter: true,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			transport := roundTripperFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: tt.body}, nil
			})
			c, err := New("http://localhost", WithAPIKey(testAPIKey), WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := CallRaw(t.Context(), c, http.MethodGet, "/audio", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if _, ok := resp.Body.(io.WriterTo); ok != tt.wantWriter {
				t.Errorf("response body implements io.WriterTo = %t, want %t", ok, tt.wantWriter)
			}
			var dst strings.Builder
			n, err := io.Copy(&dst, resp.Body)
			if n != int64(len(payload)) || dst.String() != payload {
				t.Errorf("copied %d bytes %q, want %q", n, dst.String(), payload)
			}
			if !errors.Is(err, tt.wantErr) || (err != nil && err.Error() != tt.wantText) {
				t.Errorf("copy error = %v, want %q with original cause %t", err, tt.wantText, tt.wantErr != nil)
			}
		})
	}
}
