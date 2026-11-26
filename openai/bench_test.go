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

package openai_test

import (
	"bytes"
	json "encoding/json/v2"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/openai"
)

// chatChunkSampleIndex selects the first generated token delta chunk with logprobs from stream.resp.sse.
const chatChunkSampleIndex = 3

func BenchmarkChatCompletionChunkUnmarshal(b *testing.B) {
	streamData := jsontest.ReadFile(b, "testdata/chat_completions/stream.resp.sse")
	events := jsontest.EventData(b, streamData)
	if len(events) <= chatChunkSampleIndex {
		b.Fatalf("expected at least %d events, got %d", chatChunkSampleIndex+1, len(events))
	}
	payload := events[chatChunkSampleIndex].Data

	var chunk openai.ChatCompletionChunk
	for b.Loop() {
		chunk = openai.ChatCompletionChunk{}
		if err := json.Unmarshal(payload, &chunk, json.Deterministic(true)); err != nil {
			b.Fatal(err)
		}
	}
	if chunk.ID == "" {
		b.Fatal("empty chunk id")
	}
}

func BenchmarkChatCompletionChunkStream(b *testing.B) {
	streamData := jsontest.ReadFile(b, "testdata/chat_completions/stream.resp.sse")
	events := jsontest.EventData(b, streamData)
	if len(events) <= chatChunkSampleIndex {
		b.Fatalf("expected at least %d events, got %d", chatChunkSampleIndex+1, len(events))
	}
	singleSSE := make([]byte, 0, len("data: \n\ndata: [DONE]\n\n")+len(events[chatChunkSampleIndex].Data))
	singleSSE = append(singleSSE, "data: "...)
	singleSSE = append(singleSSE, events[chatChunkSampleIndex].Data...)
	singleSSE = append(singleSSE, "\n\ndata: [DONE]\n\n"...)

	var r bytes.Reader
	var last *openai.ChatCompletionChunk
	for b.Loop() {
		r.Reset(singleSSE)
		got := 0
		for chunk, err := range llama.DecodeStream[openai.ChatCompletionChunk](&r, 0) {
			if err != nil {
				b.Fatal(err)
			}
			last = chunk
			got++
		}
		if got != 1 {
			b.Fatalf("decoded %d events, want 1", got)
		}
	}
	if last == nil || last.ID == "" {
		b.Fatal("empty chunk")
	}
	b.ReportMetric(1, "events/op")
}

func BenchmarkChatCompletionStreamDecode(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/chat_completions/stream.resp.sse")
	wantEvents := len(jsontest.EventData(b, data))
	if wantEvents == 0 {
		b.Fatal("empty chat stream fixture")
	}

	var r bytes.Reader
	var last *openai.ChatCompletionChunk
	for b.Loop() {
		r.Reset(data)
		got := 0
		for chunk, err := range llama.DecodeStream[openai.ChatCompletionChunk](&r, 0) {
			if err != nil {
				b.Fatal(err)
			}
			last = chunk
			got++
		}
		if got != wantEvents {
			b.Fatalf("decoded %d events, want %d", got, wantEvents)
		}
	}
	if last == nil {
		b.Fatal("nil final chunk")
	}
	b.ReportMetric(float64(wantEvents), "events/op")
}

func BenchmarkChatCompletionStreamUnmarshalBaseline(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/chat_completions/stream.resp.sse")
	events := jsontest.EventData(b, data)
	if len(events) == 0 {
		b.Fatal("empty chat stream fixture")
	}

	var chunk openai.ChatCompletionChunk
	for b.Loop() {
		for i := range events {
			chunk = openai.ChatCompletionChunk{}
			if err := json.Unmarshal(events[i].Data, &chunk, json.Deterministic(true)); err != nil {
				b.Fatal(err)
			}
		}
	}
	if chunk.ID == "" {
		b.Fatal("empty final chunk id")
	}
	b.ReportMetric(float64(len(events)), "events/op")
}

func BenchmarkChatCompletionStreamUnmarshalFreshPointerBaseline(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/chat_completions/stream.resp.sse")
	events := jsontest.EventData(b, data)
	var last *openai.ChatCompletionChunk
	for b.Loop() {
		for _, event := range events {
			last = new(openai.ChatCompletionChunk)
			if err := json.Unmarshal(event.Data, last, json.Deterministic(true)); err != nil {
				b.Fatal(err)
			}
		}
	}
	if last == nil || last.ID == "" {
		b.Fatal("empty final chunk")
	}
	b.ReportMetric(float64(len(events)), "events/op")
}

func BenchmarkResponsesStreamDecode(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/responses/stream.resp.sse")
	wantEvents := len(jsontest.EventData(b, data))
	if wantEvents == 0 {
		b.Fatal("empty responses stream fixture")
	}

	var r bytes.Reader
	var last *openai.ResponseEvent
	for b.Loop() {
		r.Reset(data)
		got := 0
		for ev, err := range llama.DecodeStream[openai.ResponseEvent](&r, 0) {
			if err != nil {
				b.Fatal(err)
			}
			last = ev
			got++
		}
		if got != wantEvents {
			b.Fatalf("decoded %d events, want %d", got, wantEvents)
		}
	}
	if last == nil {
		b.Fatal("nil final response event")
	}
	b.ReportMetric(float64(wantEvents), "events/op")
}

func BenchmarkEmbeddingResponseDecode(b *testing.B) {
	cases := []struct {
		name    string
		fixture string
	}{
		{name: "float", fixture: "testdata/embeddings/float.resp.json"},
		{name: "base64", fixture: "testdata/embeddings/base64.resp.json"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			data := jsontest.ReadFile(b, tc.fixture)
			var resp openai.EmbeddingResponse
			for b.Loop() {
				resp = openai.EmbeddingResponse{}
				if err := json.Unmarshal(data, &resp, json.Deterministic(true)); err != nil {
					b.Fatal(err)
				}
			}
			if len(resp.Data) == 0 {
				b.Fatal("empty embedding response")
			}
		})
	}
}

func BenchmarkChatCompletionRequestWithToolsEncode(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/chat_completions/tool.req.json")
	var req openai.ChatCompletionRequest
	if err := json.Unmarshal(data, &req, json.Deterministic(true)); err != nil {
		b.Fatal(err)
	}

	var out []byte
	for b.Loop() {
		var err error
		out, err = json.Marshal(&req, json.Deterministic(true))
		if err != nil {
			b.Fatal(err)
		}
	}
	if len(out) == 0 {
		b.Fatal("empty encoded request")
	}
}
