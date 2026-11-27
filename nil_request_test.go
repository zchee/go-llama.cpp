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
	"context"
	"iter"
	"net/http/httptrace"
	"testing"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestNilRequiredRequest(t *testing.T) {
	base, err := New("http://127.0.0.1:1")
	if err != nil {
		t.Fatal(err)
	}
	c := base
	ctx := func(t *testing.T) context.Context {
		return httptrace.WithClientTrace(t.Context(), &httptrace.ClientTrace{GetConn: func(hostPort string) {
			t.Errorf("unexpected HTTP connection attempt to %s for a nil request", hostPort)
		}})
	}
	requestError := func(_ any, err error) error { return err }
	tests := map[string]struct {
		call func(*testing.T) error
		want string
	}{
		"error: Completion":      {func(t *testing.T) error { return requestError(c.Completion(ctx(t), nil)) }, "llama: nil completion request"},
		"error: CompletionBatch": {func(t *testing.T) error { return requestError(c.CompletionBatch(ctx(t), nil)) }, "llama: nil completion batch request"},
		"error: CompletionStream": {func(t *testing.T) error {
			return jsontest.SingleErrorStream(t, func() iter.Seq2[*CompletionResponse, error] {
				return c.CompletionStream(ctx(t), nil)
			}, ErrStreamConsumed)
		}, "llama: nil completion stream request"},
		"error: Infill":      {func(t *testing.T) error { return requestError(c.Infill(ctx(t), nil)) }, "llama: nil infill request"},
		"error: InfillBatch": {func(t *testing.T) error { return requestError(c.InfillBatch(ctx(t), nil)) }, "llama: nil infill batch request"},
		"error: InfillStream": {func(t *testing.T) error {
			return jsontest.SingleErrorStream(t, func() iter.Seq2[*CompletionResponse, error] {
				return c.InfillStream(ctx(t), nil)
			}, ErrStreamConsumed)
		}, "llama: nil infill stream request"},
		"error: ApplyTemplate": {func(t *testing.T) error { return requestError(c.ApplyTemplate(ctx(t), nil)) }, "llama: nil apply template request"},
		"error: Detokenize":    {func(t *testing.T) error { return requestError(c.Detokenize(ctx(t), nil)) }, "llama: nil detokenize request"},
		"error: DownloadModel": {func(t *testing.T) error { return requestError(c.DownloadModel(ctx(t), nil)) }, "llama: nil download model request"},
		"error: Embedding":     {func(t *testing.T) error { return requestError(c.Embedding(ctx(t), nil)) }, "llama: nil embedding request"},
		"error: Embeddings":    {func(t *testing.T) error { return requestError(c.Embeddings(ctx(t), nil)) }, "llama: nil embeddings request"},
		"error: LoadModel":     {func(t *testing.T) error { return requestError(c.LoadModel(ctx(t), nil)) }, "llama: nil load model request"},
		"error: Rerank":        {func(t *testing.T) error { return requestError(c.Rerank(ctx(t), nil)) }, "llama: nil rerank request"},
		"error: RerankTEI":     {func(t *testing.T) error { return requestError(c.RerankTEI(ctx(t), nil)) }, "llama: nil rerank TEI request"},
		"error: RestoreSlot":   {func(t *testing.T) error { return requestError(c.RestoreSlot(ctx(t), 0, nil)) }, "llama: nil restore slot request"},
		"error: SaveSlot":      {func(t *testing.T) error { return requestError(c.SaveSlot(ctx(t), 0, nil)) }, "llama: nil save slot request"},
		"error: SetProps":      {func(t *testing.T) error { return requestError(c.SetProps(ctx(t), nil)) }, "llama: nil set props request"},
		"error: Tokenize":      {func(t *testing.T) error { return requestError(c.Tokenize(ctx(t), nil)) }, "llama: nil tokenize request"},
		"error: UnloadModel":   {func(t *testing.T) error { return requestError(c.UnloadModel(ctx(t), nil)) }, "llama: nil unload model request"},
		"error: TTS":           {func(t *testing.T) error { return requestError(c.TTS(ctx(t), nil)) }, "llama: nil TTS request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := tt.call(t); got == nil || got.Error() != tt.want {
				t.Errorf("error = %v, want %q", got, tt.want)
			}
		})
	}
}
