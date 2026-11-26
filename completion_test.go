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
	"io"
	"net/http"
	"net/url"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func completionCheckWire[T any](t *testing.T, data []byte) {
	t.Helper()
	jsontest.RoundTrip[T](t, data)
	jsontest.InjectUnknown[T](t, data)
	jsontest.NoUnknown[T](t, data, nil)
	jsontest.CheckFallbacks[T](t)
}

func TestCompletionResponseFixtures(t *testing.T) {
	tests := map[string]struct {
		path  string
		check func(*testing.T, []byte)
	}{
		"success: log probabilities and numeric bias": {"completion/basic", completionCheckWire[CompletionResponse]},
		"success: token ban encoded as null":          {"completion/ban", completionCheckWire[CompletionResponse]},
		"success: resolved grammar parameters":        {"completion/grammar", completionCheckWire[CompletionResponse]},
		"success: multiple prompts":                   {"completion/batch", completionCheckWire[[]*CompletionResponse]},
		"success: sibling completions":                {"completion/children", completionCheckWire[[]*CompletionResponse]},
		"success: infill":                             {"infill/basic", completionCheckWire[CompletionResponse]},
		"success: sibling infills":                    {"infill/children", completionCheckWire[[]*CompletionResponse]},
		"success: server properties":                  {"props/get", completionCheckWire[PropsResponse]},
		"success: router properties":                  {"props/router", completionCheckWire[PropsResponse]},
		"success: property acknowledgement":           {"props/set", completionCheckWire[PropsSetResponse]},
		"success: fresh slots":                        {"slots/idle", completionCheckWire[[]Slot]},
		"success: used slots":                         {"slots/used", completionCheckWire[[]Slot]},
		"success: saved slot":                         {"slots/save", completionCheckWire[SlotSaveResponse]},
		"success: restored slot":                      {"slots/restore", completionCheckWire[SlotRestoreResponse]},
		"success: erased slot":                        {"slots/erase", completionCheckWire[SlotEraseResponse]},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) { tt.check(t, jsontest.ReadFile(t, "testdata/"+tt.path+".resp.json")) })
	}
}

func TestCompletionProjectedFields(t *testing.T) {
	data := jsontest.ReadFile(t, "testdata/completion/projection.resp.json")
	got := jsontest.RoundTrip[CompletionResponse](t, data)
	jsontest.InjectUnknown[CompletionResponse](t, data)
	jsontest.NoUnknown[CompletionResponse](t, data, map[string]string{"": "response_fields moves arbitrary JSON paths to the top level"})
	if diff := gocmp.Diff(map[string]jsontext.Value{"generation_settings/n_predict": jsontext.Value("1")}, got.Extra); diff != "" {
		t.Errorf("projected fields (-want +got):\n%s", diff)
	}
	if got.Model != nil || got.Content == nil {
		t.Errorf("projection synthesized missing fields or lost content: %+v", got)
	}
}

func TestCompletionStreamFixtures(t *testing.T) {
	tests := map[string]struct {
		path         string
		wantProgress bool
	}{
		"success: progress and probabilities": {"completion/stream", true},
		"success: infill tokens":              {"infill/stream", false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/"+tt.path+".resp.sse")
			events := jsontest.EventData(t, data)
			var frames, final int
			var progress, probs bool
			for frame, err := range DecodeStream[CompletionResponse](bytes.NewReader(data), 0) {
				if err != nil {
					t.Fatal(err)
				}
				if frames >= len(events) || final != 0 {
					t.Fatal("unexpected frame after final result")
				}
				completionCheckWire[CompletionResponse](t, events[frames].Data)
				frames++
				if frame.Stop == nil {
					t.Fatal("missing stop flag")
				}
				if *frame.Stop {
					final++
				}
				progress = progress || frame.PromptProgress != nil
				if len(frame.CompletionProbabilities) > 0 {
					p := frame.CompletionProbabilities[0]
					probs = probs || (p.Prob != nil && len(p.TopProbs) > 0 && p.Logprob == nil)
				}
			}
			if frames != len(events) || final != 1 || progress != tt.wantProgress || probs != tt.wantProgress {
				t.Fatalf("got %d/%d events, final=%d, progress=%t, probabilities=%t", frames, len(events), final, progress, probs)
			}
		})
	}
}

func completionRequestFixture[T any](t *testing.T, data []byte) any {
	t.Helper()
	completionCheckWire[T](t, data)
	var v T
	if err := json.Unmarshal(data, &v); err != nil {
		t.Fatal(err)
	}
	return &v
}

func TestCompletionNativeRequests(t *testing.T) {
	c, err := New("http://localhost:8080")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		fixture, path string
		read          func(*testing.T, []byte) any
		opts          []RequestOption
	}{
		"success: text":            {"completion/basic", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: batch":           {"completion/batch", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: tokens":          {"completion/children", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: projection":      {"completion/projection", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: stream":          {"completion/stream", "/completion", completionRequestFixture[CompletionRequest], nil},
		"error: media":             {"completion/media", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: grammar":         {"completion/grammar", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: ban":             {"completion/ban", "/completion", completionRequestFixture[CompletionRequest], nil},
		"success: infill contexts": {"infill/basic", "/infill", completionRequestFixture[InfillRequest], nil},
		"success: infill tokens":   {"infill/stream", "/infill", completionRequestFixture[InfillRequest], nil},
		"success: infill siblings": {"infill/children", "/infill", completionRequestFixture[InfillRequest], nil},
		"success: read properties": {"", "/props", nil, nil},
		"success: post properties": {"props/set", "/props", completionRequestFixture[PropsSetRequest], nil},
		"success: slots":           {"", "/slots", nil, []RequestOption{WithFailOnNoSlot()}},
		"success: save":            {"slots/save", "/slots/0?action=save", completionRequestFixture[SlotFileRequest], nil},
		"success: restore":         {"slots/restore", "/slots/0?action=restore", completionRequestFixture[SlotFileRequest], nil},
		"success: erase":           {"", "/slots/0?action=erase", nil, nil},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			method := http.MethodPost
			var body any
			var want jsontext.Value
			if tt.read != nil {
				want = jsontest.ReadFile(t, "testdata/"+tt.fixture+".req.json")
				body = tt.read(t, want)
			} else if tt.path == "/slots" || tt.path == "/props" {
				method = http.MethodGet
			}
			req, err := c.NewRequest(t.Context(), method, tt.path, body, tt.opts...)
			if err != nil {
				t.Fatal(err)
			}
			u, err := url.Parse(tt.path)
			if err != nil {
				t.Fatal(err)
			}
			q := u.Query()
			if tt.path == "/slots" {
				q.Set("fail_on_no_slot", "1")
			}
			if req.Method != method || req.URL.Path != u.Path || req.URL.RawQuery != q.Encode() {
				t.Fatalf("built %s %s, want %s %s %v", req.Method, req.URL, method, u.Path, q)
			}
			if want == nil {
				if req.Body != nil {
					t.Fatal("unexpected request body")
				}
				return
			}
			defer req.Body.Close()
			data, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			got := jsontext.Value(data)
			if err := got.Canonicalize(); err != nil {
				t.Fatal(err)
			}
			if err := want.Canonicalize(); err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(string(want), string(got)); diff != "" {
				t.Errorf("body (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCompletionNativeErrors(t *testing.T) {
	tests := map[string]struct {
		path   string
		status int
		kind   ErrorType
	}{
		"error: media unsupported":        {"completion/media", 400, ErrorTypeInvalidRequest},
		"error: no infill tokens":         {"infill/unsupported", 501, ErrorTypeNotSupported},
		"error: changing props disabled":  {"props/disabled", 501, ErrorTypeNotSupported},
		"error: slot inspection disabled": {"slots/disabled", 501, ErrorTypeNotSupported},
		"error: slot actions disabled":    {"slots/action_disabled", 501, ErrorTypeNotSupported},
		"error: missing cache":            {"slots/missing", 400, ErrorTypeInvalidRequest},
		"error: invalid cache filename":   {"slots/invalid_filename", 400, ErrorTypeInvalidRequest},
		"error: all slots busy":           {"slots/busy", 503, ErrorTypeUnavailable},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/"+tt.path+".resp.json")
			e := ReadAPIError(&http.Response{StatusCode: tt.status, Body: io.NopCloser(bytes.NewReader(data))})
			if e.Code != tt.status || e.Type != tt.kind || e.Message == "" {
				t.Fatalf("server error = %+v, want %d %s", e, tt.status, tt.kind)
			}
		})
	}
}

func FuzzCompletionResults(f *testing.F) {
	for _, name := range []string{"basic", "batch", "projection", "children"} {
		f.Add(jsontest.ReadFile(f, "testdata/completion/"+name+".resp.json"))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var first, second completionResults
		if json.Unmarshal(data, &first) != nil {
			return
		}
		out, err := json.Marshal(first, jsonOptions)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(out, &second); err != nil {
			t.Fatal(err)
		}
		again, err := json.Marshal(second, jsonOptions)
		if err != nil || !bytes.Equal(out, again) {
			t.Fatalf("unstable completion results: %s -> %s (%v)", out, again, err)
		}
	})
}
