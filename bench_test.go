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

package llama_test

import (
	"bytes"
	json "encoding/json/v2"
	"net/http"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func BenchmarkCompletionStreamDecode(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/completion/stream.resp.sse")
	wantEvents := len(jsontest.EventData(b, data))
	if wantEvents == 0 {
		b.Fatal("empty completion stream fixture")
	}

	var r bytes.Reader
	var last *llama.CompletionResponse
	for b.Loop() {
		r.Reset(data)
		got := 0
		for resp, err := range llama.DecodeStream[llama.CompletionResponse](&r, 0) {
			if err != nil {
				b.Fatal(err)
			}
			last = resp
			got++
		}
		if got != wantEvents {
			b.Fatalf("decoded %d events, want %d", got, wantEvents)
		}
	}
	if last == nil {
		b.Fatal("nil final completion event")
	}
	b.ReportMetric(float64(wantEvents), "events/op")
}

func BenchmarkPromptEncode(b *testing.B) {
	tests := map[string]struct {
		fixture  string
		wantKind llama.PromptKind
	}{
		"text":   {fixture: "testdata/completion/basic.req.json", wantKind: llama.PromptKindText},
		"tokens": {fixture: "testdata/completion/children.req.json", wantKind: llama.PromptKindTokens},
		"mixed":  {fixture: "testdata/completion/projection.req.json", wantKind: llama.PromptKindMixed},
		"object": {fixture: "testdata/completion/media.req.json", wantKind: llama.PromptKindObject},
		"batch":  {fixture: "testdata/completion/batch.req.json", wantKind: llama.PromptKindBatch},
	}

	for name, tc := range tests {
		b.Run(name, func(b *testing.B) {
			data := jsontest.ReadFile(b, tc.fixture)
			var req llama.CompletionRequest
			if err := json.Unmarshal(data, &req, json.Deterministic(true)); err != nil {
				b.Fatal(err)
			}
			prompt := req.Prompt
			if prompt.Kind() != tc.wantKind {
				b.Fatalf("prompt kind = %d, want %d", prompt.Kind(), tc.wantKind)
			}

			var out []byte
			for b.Loop() {
				var err error
				out, err = json.Marshal(&prompt, json.Deterministic(true))
				if err != nil {
					b.Fatal(err)
				}
			}
			if len(out) == 0 {
				b.Fatal("empty encoded prompt")
			}
		})
	}
}

func BenchmarkClientNewRequest(b *testing.B) {
	c, err := llama.New("http://127.0.0.1:8080", llama.WithAPIKey("bench-api-key"))
	if err != nil {
		b.Fatal(err)
	}
	data := jsontest.ReadFile(b, "testdata/completion/basic.req.json")
	var compReq llama.CompletionRequest
	if err := json.Unmarshal(data, &compReq, json.Deterministic(true)); err != nil {
		b.Fatal(err)
	}

	ctx := b.Context()

	b.Run("get", func(b *testing.B) {
		var req *http.Request
		for b.Loop() {
			var err error
			req, err = c.NewRequest(ctx, http.MethodGet, "/slots", nil, llama.WithModelQuery("stories"))
			if err != nil {
				b.Fatal(err)
			}
		}
		if req == nil {
			b.Fatal("nil request")
		}
	})

	b.Run("post_json", func(b *testing.B) {
		var req *http.Request
		for b.Loop() {
			var err error
			req, err = c.NewRequest(ctx, http.MethodPost, "/completion", &compReq, llama.WithModelQuery("stories"))
			if err != nil {
				b.Fatal(err)
			}
		}
		if req == nil {
			b.Fatal("nil request")
		}
	})
}
