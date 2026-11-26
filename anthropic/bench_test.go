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

package anthropic_test

import (
	"bytes"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/anthropic"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func BenchmarkMessagesStreamDecode(b *testing.B) {
	data := jsontest.ReadFile(b, "testdata/messages/stream.resp.sse")
	wantEvents := len(jsontest.EventData(b, data))
	if wantEvents == 0 {
		b.Fatal("empty anthropic stream fixture")
	}

	var r bytes.Reader
	var last *anthropic.MessageEvent
	for b.Loop() {
		r.Reset(data)
		got := 0
		for ev, err := range llama.DecodeStream[anthropic.MessageEvent](&r, 0) {
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
		b.Fatal("nil final message event")
	}
	b.ReportMetric(float64(wantEvents), "events/op")
}
