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
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func modelsWire[T any](t *testing.T, name string) *T {
	t.Helper()
	data := jsontest.ReadFile(t, "testdata/models/"+name+".json")
	v := jsontest.RoundTrip[T](t, data)
	jsontest.NoUnknown[T](t, data, nil)
	jsontest.InjectUnknown[T](t, data)
	jsontest.CheckFallbacks[T](t)
	return v
}

func TestModelsWire(t *testing.T) {
	tests := map[string]struct{ check func(*testing.T) }{
		"success: single model and compatibility listing": {func(t *testing.T) {
			got := modelsWire[ModelList](t, "single.resp")
			if len(got.Data) != 1 || got.Data[0].ID != "stories" || got.Data[0].Meta == nil || got.Data[0].Meta.NVocab != 512 || len(got.Models) != 1 || got.Models[0].Details.Format != "gguf" {
				t.Errorf("unexpected single-model listing: %+v", got)
			}
		}},
		"success: router lists": {func(t *testing.T) {
			for _, name := range []string{"empty", "incomplete", "downloading", "unloaded", "loading", "loaded", "sleeping", "failed", "cached"} {
				list := modelsWire[ModelList](t, "list-"+name+".resp")
				if name == "empty" {
					if len(list.Data) != 0 {
						t.Errorf("expected empty list: %+v", list)
					}
					continue
				}
				if len(list.Data) == 0 {
					t.Fatalf("%s list has no models", name)
				}
				for i := range list.Data {
					s := list.Data[i].Status
					if s == nil || s.Args == nil {
						t.Fatalf("%s: missing status or argument list", name)
					}
					if name != "incomplete" && (len(s.Args) == 0 || s.Args[0] == "") {
						t.Errorf("%s: missing executable argument", name)
					}
					want := ModelState(name)
					if name == "failed" || name == "cached" {
						want = ModelUnloaded
					}
					if s.Value != want {
						t.Errorf("%s: status = %q, want %q", name, s.Value, want)
					}
					if name == "failed" && (s.Failed == nil || !*s.Failed || s.ExitCode == nil || *s.ExitCode == 0) {
						t.Errorf("unreported child failure: %+v", s)
					}
				}
			}
		}},
		"success: load or unload request": {func(t *testing.T) {
			for _, f := range []string{"load", "unload", "not-found", "not-running"} {
				modelsWire[ModelRequest](t, f+".req")
			}
		}},
		"success: download request": {func(t *testing.T) {
			got := modelsWire[DownloadModelRequest](t, "download.req")
			if got.Sidecars == nil || len(got.Sidecars) != 0 {
				t.Errorf("explicitly empty sidecars lost: %#v", got.Sidecars)
			}
			modelsWire[DownloadModelRequest](t, "bad-sidecars.req")
		}},
		"success: operation responses": {func(t *testing.T) {
			for _, f := range []string{"load", "unload", "download", "delete", "delete-sidecar"} {
				res := modelsWire[ModelActionResponse](t, f+".resp")
				if res.Success != (f != "delete-sidecar") {
					t.Errorf("%s: success = %t", f, res.Success)
				}
			}
		}},
	}
	for name, tt := range tests {
		t.Run(name, tt.check)
	}
}

func TestModelsErrors(t *testing.T) {
	tests := map[string]struct {
		file   string
		status int
	}{"error: unknown model": {"not-found", 404}, "error: model not running": {"not-running", 400}, "error: invalid download sidecars": {"bad-sidecars", 400}, "error: invalid deletion sidecar": {"invalid-delete", 400}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/models/"+tt.file+".resp.json")
			e := ReadAPIError(&http.Response{StatusCode: tt.status, Body: io.NopCloser(bytes.NewReader(data))})
			if e.Code != tt.status || e.Message == "" || e.Type == "" {
				t.Errorf("unexpected decoded error: %+v", e)
			}
		})
	}
}

func TestModelsEventsWire(t *testing.T) {
	tests := map[string]struct{ file string }{
		"success: router lifecycle":         {"router"},
		"success: single-model replay":      {"single"},
		"success: paused download":          {"pause"},
		"success: child exited during load": {"failed"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			stream := jsontest.ReadFile(t, "testdata/models/"+tt.file+".resp.sse")
			frames := jsontest.EventData(t, stream)
			i := 0
			for event, err := range DecodeStream[ModelEvent](bytes.NewReader(stream), 0) {
				if err != nil {
					t.Fatal(err)
				}
				if i >= len(frames) {
					t.Fatal("unexpected extra event")
				}
				data := frames[i].Data
				i++
				got := jsontest.RoundTrip[ModelEvent](t, data)
				jsontest.NoUnknown[ModelEvent](t, data, nil)
				if got.Event == "download_progress" { // The keys of progress are file URLs, not object members.
					jsontest.InjectUnknown[ModelEvent](t, data, jsontest.SkipObjects("/data/progress"))
				} else {
					jsontest.InjectUnknown[ModelEvent](t, data)
				}
				if diff := gocmp.Diff(got, event); diff != "" {
					t.Fatal(diff)
				}
				switch got.Event {
				case "download_progress":
					if d, ok := got.Data.(*ModelDownloadProgress); !ok || len(d.Progress) == 0 {
						t.Errorf("download_progress payload: %#v", got.Data)
					}
				case "status_change", "model_status":
					if d, ok := got.Data.(*ModelStatusChange); !ok || d.Status == "" {
						t.Errorf("status payload: %#v", got.Data)
					}
				case "model_memory":
					if d, ok := got.Data.(*ModelMemory); !ok || d.Status == "" {
						t.Errorf("memory payload: %#v", got.Data)
					}
				case "download_failed":
					if d, ok := got.Data.(*ModelDownloadFailure); !ok || !d.Stopped {
						t.Errorf("expected a paused download: %#v", got.Data)
					}
				}
			}
			if i == 0 || i != len(frames) {
				t.Errorf("decoded %d of %d events", i, len(frames))
			}
		})
	}
	jsontest.CheckFallbacks[ModelEvent](t)
	jsontest.CheckFallbacks[ModelStatusChange](t)
	jsontest.CheckFallbacks[ModelDownloadProgress](t)
	jsontest.CheckFallbacks[ModelMemory](t)
	jsontest.CheckFallbacks[ModelDownloadFailure](t)
}

func TestModelsUnknownEventData(t *testing.T) {
	tests := map[string]struct{ data string }{
		"success: future event with array data before discriminator": {`{"data":[1,{"custom":true}],"event":"future","model":"demo"}`},
		"success: explicit null stays distinct from absent data":     {`{"data":null,"event":"model_status","model":"demo"}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			v := jsontest.RoundTrip[ModelEvent](t, []byte(tt.data))
			if _, ok := v.Data.(*ModelRawEventData); !ok {
				t.Fatalf("unexpected payload type: %T", v.Data)
			}
		})
	}
}

func FuzzModelsEvent(f *testing.F) {
	for _, name := range []string{"router", "single", "pause", "failed"} {
		for _, e := range jsontest.EventData(f, jsontest.ReadFile(f, "testdata/models/"+name+".resp.sse")) {
			f.Add(e.Data)
		}
	}
	f.Fuzz(fuzzModelEncoding[ModelEvent])
}

func TestModelsRejectAmbiguousSidecar(t *testing.T) {
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct{ opts []RequestOption }{
		"error: empty sidecar could delete entire model":      {[]RequestOption{WithSidecar("")}},
		"error: repeated sidecar could delete the wrong file": {[]RequestOption{WithSidecar("mmproj"), WithSidecar("mtp")}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, err := c.DeleteModel(ctx, "org/model", tt.opts...)
			if err == nil || !strings.Contains(err.Error(), "single non-empty sidecar") {
				t.Errorf("expected sidecar rejection without a request; got %v", err)
			}
		})
	}
}

func TestModelsRejectEmptyDeleteName(t *testing.T) {
	sent := false
	c, err := New("http://localhost", WithHTTPClient(&http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		sent = true
		return nil, fmt.Errorf("unexpected transport call")
	})}))
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		model string
		opts  []RequestOption
		want  string
	}{
		"error: empty model name":       {model: "", want: "non-empty model name"},
		"error: empty name and sidecar": {model: "", opts: []RequestOption{WithSidecar("")}, want: "non-empty model name"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			sent = false
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, err := c.DeleteModel(ctx, tt.model, tt.opts...)
			if sent {
				t.Fatal("DeleteModel sent an HTTP request for an empty model name")
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("DeleteModel error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func FuzzModelsStatus(f *testing.F) {
	lists, err := filepath.Glob("testdata/models/list-*.resp.json")
	if err != nil {
		f.Fatal(err)
	}
	for _, path := range lists {
		var models struct {
			Data []struct {
				Status jsontext.Value `json:"status"`
			} `json:"data"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, path), &models); err != nil {
			f.Fatal(err)
		}
		for _, m := range models.Data {
			f.Add([]byte(m.Status))
		}
	}
	f.Fuzz(fuzzModelEncoding[ModelStatus])
}

func fuzzModelEncoding[T any](t *testing.T, data []byte) {
	var first, second T
	if err := json.Unmarshal(data, &first); err != nil {
		return
	}
	encoded, err := json.Marshal(first, json.Deterministic(true))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded, &second); err != nil {
		t.Fatal(err)
	}
	again, err := json.Marshal(second, json.Deterministic(true))
	if err != nil || !bytes.Equal(encoded, again) {
		t.Errorf("unstable %T encoding: %s -> %s (%v)", first, encoded, again, err)
	}
}

func TestModelsNewRequest(t *testing.T) {
	c, err := New("http://localhost/prefix")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		method, path string
		body         any
		opts         []RequestOption
		query        url.Values
		fixture      string
	}{
		"success: list with rescan":     {method: "GET", path: "/models", opts: []RequestOption{WithReload()}, query: url.Values{"reload": {"1"}}},
		"success: OpenAI alias":         {method: "GET", path: "/v1/models"},
		"success: subscribe":            {method: "GET", path: "/models/sse"},
		"success: load":                 {method: "POST", path: "/models/load", body: &ModelRequest{Model: "stories260K-f32"}, fixture: "load"},
		"success: unload":               {method: "POST", path: "/models/unload", body: &ModelRequest{Model: "stories260K-f32"}, fixture: "unload"},
		"success: download no sidecars": {method: "POST", path: "/models", body: &DownloadModelRequest{Model: "ggml-org-dev/test-model-stories260K", Sidecars: []string{}}, fixture: "download"},
		"success: remove a sidecar":     {method: "DELETE", path: "/models", opts: []RequestOption{WithModelQuery("org/m:Q4"), WithSidecar("mmproj")}, query: url.Values{"model": {"org/m:Q4"}, "sidecar": {"mmproj"}}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), tt.method, tt.path, tt.body, tt.opts...)
			if err != nil {
				t.Fatal(err)
			}
			if req.Method != tt.method || req.URL.Path != "/prefix"+tt.path || req.URL.Query().Encode() != tt.query.Encode() {
				t.Errorf("unexpected request: %s %s", req.Method, req.URL)
			}
			if req.Body != nil {
				defer req.Body.Close()
			}
			if tt.fixture != "" {
				b, err := io.ReadAll(req.Body)
				if err != nil {
					t.Fatal(err)
				}
				if diff := gocmp.Diff(string(jsontest.ReadFile(t, "testdata/models/"+tt.fixture+".req.json")), string(b)); diff != "" {
					t.Error(diff)
				}
			}
		})
	}
}

func ExampleClient_ListModels() {
	client, err := New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	list, err := client.ListModels(context.Background(), WithReload())
	if err != nil {
		panic(err)
	}
	for i := range list.Data {
		fmt.Println(list.Data[i].ID)
	}
}
