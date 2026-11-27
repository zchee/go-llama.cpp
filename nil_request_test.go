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
	ctx := httptrace.WithClientTrace(t.Context(), &httptrace.ClientTrace{GetConn: func(hostPort string) {
		t.Errorf("unexpected HTTP connection attempt to %s for a nil request", hostPort)
	}})
	requestError := func(_ any, err error) error { return err }
	completionStream := func() iter.Seq2[*CompletionResponse, error] { return c.CompletionStream(ctx, nil) }
	infillStream := func() iter.Seq2[*CompletionResponse, error] { return c.InfillStream(ctx, nil) }
	tests := map[string]struct {
		got  error
		want string
	}{
		"error: Completion":       {requestError(c.Completion(ctx, nil)), "llama: nil completion request"},
		"error: CompletionBatch":  {requestError(c.CompletionBatch(ctx, nil)), "llama: nil completion batch request"},
		"error: CompletionStream": {jsontest.SingleErrorStream(t, completionStream, ErrStreamConsumed), "llama: nil completion stream request"},
		"error: Infill":           {requestError(c.Infill(ctx, nil)), "llama: nil infill request"},
		"error: InfillBatch":      {requestError(c.InfillBatch(ctx, nil)), "llama: nil infill batch request"},
		"error: InfillStream":     {jsontest.SingleErrorStream(t, infillStream, ErrStreamConsumed), "llama: nil infill stream request"},
		"error: ApplyTemplate":    {requestError(c.ApplyTemplate(ctx, nil)), "llama: nil apply template request"},
		"error: Detokenize":       {requestError(c.Detokenize(ctx, nil)), "llama: nil detokenize request"},
		"error: DownloadModel":    {requestError(c.DownloadModel(ctx, nil)), "llama: nil download model request"},
		"error: Embedding":        {requestError(c.Embedding(ctx, nil)), "llama: nil embedding request"},
		"error: Embeddings":       {requestError(c.Embeddings(ctx, nil)), "llama: nil embeddings request"},
		"error: LoadModel":        {requestError(c.LoadModel(ctx, nil)), "llama: nil load model request"},
		"error: Rerank":           {requestError(c.Rerank(ctx, nil)), "llama: nil rerank request"},
		"error: RerankTEI":        {requestError(c.RerankTEI(ctx, nil)), "llama: nil rerank TEI request"},
		"error: RestoreSlot":      {requestError(c.RestoreSlot(ctx, 0, nil)), "llama: nil restore slot request"},
		"error: SaveSlot":         {requestError(c.SaveSlot(ctx, 0, nil)), "llama: nil save slot request"},
		"error: SetProps":         {requestError(c.SetProps(ctx, nil)), "llama: nil set props request"},
		"error: Tokenize":         {requestError(c.Tokenize(ctx, nil)), "llama: nil tokenize request"},
		"error: UnloadModel":      {requestError(c.UnloadModel(ctx, nil)), "llama: nil unload model request"},
		"error: TTS":              {requestError(c.TTS(ctx, nil)), "llama: nil TTS request"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if tt.got == nil || tt.got.Error() != tt.want {
				t.Errorf("error = %v, want %q", tt.got, tt.want)
			}
		})
	}
}
