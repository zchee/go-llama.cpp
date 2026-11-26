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

package integration

import (
	"bytes"
	"crypto/sha256"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

// Each case captures one complete exchange, including SSE bytes the SDK might not consume.
type sdkExchange struct {
	base, path        string
	request, response []byte
	client            *llama.Client
	next              http.RoundTripper
}

func newSDKExchange(t *testing.T, s *harness.Server, path string) *sdkExchange {
	t.Helper()
	c, _ := s.Client(t, llama.WithAPIKey(testAPIKey))
	return &sdkExchange{base: s.URL, path: path, client: c, next: c.HTTPClient().Transport}
}

func (x *sdkExchange) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body == nil {
		return nil, fmt.Errorf("SDK sent a request without a body")
	}
	defer req.Body.Close()
	if req.Method != http.MethodPost || req.URL.Scheme+"://"+req.URL.Host != x.base || req.URL.Path != x.path || req.URL.User != nil {
		return nil, fmt.Errorf("unexpected SDK destination or method: %s %s://%s%s", req.Method, req.URL.Scheme, req.URL.Host, req.URL.Path)
	}
	if x.request != nil {
		return nil, fmt.Errorf("duplicate SDK request to %s", x.path)
	}
	var err error
	x.request, err = io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	copyReq := req.Clone(req.Context())
	copyReq.Body = io.NopCloser(bytes.NewReader(x.request))
	res, err := x.next.RoundTrip(copyReq)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	x.response, err = io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	res.Body = io.NopCloser(bytes.NewReader(x.response))
	return res, nil
}

// Compare the actual SDK POST (including any stream flag added by its service method), then submit the unmodified library encoding.
func (x *sdkExchange) checkRequest(t *testing.T, req any, nativeOnly ...string) {
	t.Helper()
	body := sdkJSON(t, req)
	var common map[string]jsontext.Value
	if err := json.Unmarshal(body, &common); err != nil {
		t.Fatal(err)
	}
	for _, key := range nativeOnly {
		if _, ok := common[key]; !ok {
			t.Fatalf("expected llama.cpp request member %s", key)
		}
		delete(common, key)
	}
	if !equalSDKJSON(t, sdkJSON(t, common), x.request) {
		t.Errorf("request mismatch\nSDK: %s\nlibrary: %s", x.request, body)
	}
	res, err := llama.CallRaw(t.Context(), x.client, http.MethodPost, x.path, body)
	if err != nil {
		t.Fatalf("library POST %s: %v\nbody: %s", x.path, err, body)
	}
	defer res.Body.Close()
	if strings.HasPrefix(res.Header.Get("Content-Type"), "text/event-stream") {
		n := 0
		for _, err := range llama.DecodeStream[jsontext.Value](res.Body, 0) {
			if err != nil {
				t.Fatal(err)
			}
			n++
		}
		if n == 0 {
			t.Fatal("library POST returned an empty stream")
		}
		return
	}
	var answer jsontext.Value
	if err := json.UnmarshalRead(res.Body, &answer); err != nil {
		t.Fatalf("library POST response: %v", err)
	}
}

func equalSDKJSON(t *testing.T, a, b jsontext.Value) bool {
	t.Helper()
	a, b = a.Clone(), b.Clone()
	for _, v := range []*jsontext.Value{&a, &b} {
		if err := v.Canonicalize(jsontext.CanonicalizeRawInts(false)); err != nil {
			t.Fatal(err)
		}
	}
	return bytes.Equal(a, b)
}

func sdkJSON(t *testing.T, v any) jsontext.Value {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("encoding %T: %v", v, err)
	}
	return b
}

func (x *sdkExchange) checkResponse[T any](t *testing.T, sdk any) {
	t.Helper()
	local := jsontest.RoundTrip[T](t, x.response)
	checkSDKFields(t, "", sdkJSON(t, local), reflect.Indirect(reflect.ValueOf(sdk)))
}

// Walk only members sent on the wire: the generated SDK structs also expose zero-valued fields for absent members.
// Both SDKs use JSON.<Field> metadata to distinguish a valid zero, an explicit null and an invalid value.
func checkSDKFields(t *testing.T, path string, local jsontext.Value, sdk reflect.Value) {
	t.Helper()
	switch {
	case local.Kind() == '{' && sdk.Kind() == reflect.Struct:
		var members map[string]jsontext.Value
		if err := json.Unmarshal(local, &members); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < sdk.NumField(); i++ {
			f := sdk.Type().Field(i)
			key, _, _ := strings.Cut(f.Tag.Get("json"), ",")
			wire, ok := members[key]
			if !ok {
				continue
			}
			delete(members, key)
			p := path + "/" + key
			metadata := sdk.FieldByName("JSON")
			if !metadata.IsValid() || !metadata.FieldByName(f.Name).IsValid() {
				t.Fatalf("%s: SDK type %s no longer exposes field presence for %s; wire: %s", p, sdk.Type(), f.Name, local)
			}
			meta := metadata.FieldByName(f.Name).Interface().(interface {
				Valid() bool
				Raw() string
			})
			if !meta.Valid() {
				if meta.Raw() != "null" || wire.Kind() != 'n' {
					t.Errorf("%s: SDK field absent or invalid (%q); library: %s\nobject: %s", p, meta.Raw(), wire, local)
				}
				continue
			}
			checkSDKFields(t, p, wire, sdk.Field(i))
		}
		for k := range members {
			t.Logf("not typed by SDK %s: %s/%s", sdk.Type(), path, k)
		}
	case local.Kind() == '[' && sdk.Kind() == reflect.Slice:
		var items []jsontext.Value
		if err := json.Unmarshal(local, &items); err != nil {
			t.Fatal(err)
		}
		if len(items) != sdk.Len() {
			t.Errorf("%s: SDK length %d; library: %s", path, sdk.Len(), local)
			return
		}
		for i, item := range items {
			checkSDKFields(t, fmt.Sprintf("%s/%d", path, i), item, sdk.Index(i))
		}
	default:
		if v := sdkJSON(t, sdk.Interface()); !equalSDKJSON(t, local, v) {
			t.Errorf("%s: SDK: %s; library: %s", path, v, local)
		}
	}
}

type sdkStream[S any] interface {
	Next() bool
	Current() S
	Err() error
	Close() error
}

func (x *sdkExchange) checkStream[L, S any](t *testing.T, stream sdkStream[S]) {
	t.Helper()
	defer stream.Close()
	frames := jsontest.EventData(t, x.response)
	n := 0
	for local, err := range llama.DecodeStream[L](bytes.NewReader(x.response), 0) {
		if err != nil {
			t.Fatalf("library stream: %v\nwire: %s", err, x.response)
		}
		if !stream.Next() || n >= len(frames) {
			t.Fatalf("SDK stream ended at event %d: %v\nwire: %s", n, stream.Err(), x.response)
		}
		data := sdkJSON(t, local)
		if !equalSDKJSON(t, data, frames[n].Data) {
			t.Fatalf("library event %d: %s\nwire: %s", n, data, frames[n].Data)
		}
		checkSDKFields(t, fmt.Sprintf("/events/%d", n), data, reflect.ValueOf(stream.Current()))
		n++
	}
	if stream.Next() || stream.Err() != nil || n != len(frames) || n == 0 {
		t.Fatalf("SDK and library streams did not finish together: events=%d frames=%d err=%v\nwire: %s", n, len(frames), stream.Err(), x.response)
	}
}

// The tool grammar and assistant prefill need more context than the ordinary message preset.
var presetSDKTools = &serverPreset{name: "sdk-tools", options: func(t testing.TB) harness.Options {
	// llama.cpp models/templates/Qwen-Qwen3-0.6B.jinja at 171527522610996f5f0acf12334e185156ca8f44,
	// SHA-256: 87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5.
	const path = "testdata/Qwen-Qwen3-0.6B.jinja"
	data := jsontest.ReadFile(t, path)
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5" {
		t.Fatalf("%s: unexpected SHA-256 %s", path, got)
	}
	templatePath, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	opts := presetAnthropic.options(t)
	opts.Args = []string{"--alias", "stories", "--ctx-size", "2048", "--parallel", "1", "--jinja", "--seed", "42", "--chat-template-file", templatePath}
	return opts
}}
