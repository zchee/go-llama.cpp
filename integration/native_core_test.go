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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"iter"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

var slotsPreset = &serverPreset{name: "stories-single-slot", options: func(t testing.TB) harness.Options {
	return harness.Options{Model: harness.Model(t, "stories260K"), Args: []string{"--parallel", "1", "--props", "--slot-save-path", ".", "--ctx-size", "512", "--seed", "42"}}
}}

var infillPreset = &serverPreset{name: "stories-infill", options: func(t testing.TB) harness.Options {
	return harness.Options{Model: harness.Model(t, "stories260K-infill"), Args: []string{"--parallel", "2", "--ctx-size", "512", "--seed", "42"}}
}}

func TestCompletionBatchAndAlias(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	tests := map[string]struct {
		prompt llama.Prompt
		n      int
		want   int
	}{
		"success: one object":          {llama.TextPrompt("Once"), 1, 1},
		"success: several prompts":     {llama.BatchPrompt(llama.TextPrompt("Once"), llama.TextPrompt("A")), 1, 2},
		"success: sibling completions": {llama.TextPrompt("Once"), 2, 2},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req := &llama.CompletionRequest{Prompt: tt.prompt, NPredict: new(1), NCmpl: new(tt.n), Stream: new(true)}
			rs, err := c.CompletionBatch(t.Context(), req)
			if err != nil || len(rs) != tt.want {
				t.Fatalf("CompletionBatch returned %d results, want %d: %v", len(rs), tt.want, err)
			}
			if !*req.Stream {
				t.Fatal("CompletionBatch modified the caller's stream flag")
			}
			if tt.want > 1 {
				r, err := c.Completion(t.Context(), req)
				if r != nil || err == nil || !strings.Contains(err.Error(), "CompletionBatch") {
					t.Fatalf("Completion on %d results = %+v, %v", tt.want, r, err)
				}
			}
		})
	}
	r, err := llama.Call[llama.CompletionResponse](t.Context(), c, http.MethodPost, "/completions", &llama.CompletionRequest{Prompt: llama.TextPrompt("Once"), NPredict: new(1)})
	if err != nil || r.Stop == nil || !*r.Stop {
		t.Fatalf("completion alias: %+v, %v", r, err)
	}
}

func TestCompletionAndInfillStreams(t *testing.T) {
	tests := map[string]struct {
		preset *serverPreset
		stream func(*testing.T, *llama.Client) iter.Seq2[*llama.CompletionResponse, error]
	}{
		"success: native completion": {presetStories260K, func(t *testing.T, c *llama.Client) iter.Seq2[*llama.CompletionResponse, error] {
			return c.CompletionStream(t.Context(), &llama.CompletionRequest{Prompt: llama.TextPrompt("Once"), NPredict: new(2), ReturnProgress: new(true), TimingsPerToken: new(true), NProbs: new(2), PostSamplingProbs: new(true)})
		}},
		"success: code infill": {infillPreset, func(t *testing.T, c *llama.Client) iter.Seq2[*llama.CompletionResponse, error] {
			return c.InfillStream(t.Context(), &llama.InfillRequest{InputPrefix: llama.InfillInput{Text: "Once"}, InputSuffix: llama.InfillInput{Text: "end"}, NPredict: new(2), ReturnProgress: new(true), TimingsPerToken: new(true), NProbs: new(2), PostSamplingProbs: new(true)})
		}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, obs := tt.preset.server(t).Client(t)
			var progress, tokens, final int
			for chunk, err := range tt.stream(t, c) {
				if err != nil {
					t.Fatal(err)
				}
				if chunk.Stop == nil || chunk.Timings == nil || final != 0 {
					t.Fatalf("unexpected event: %+v", chunk)
				}
				switch {
				case *chunk.Stop:
					final++
				case chunk.PromptProgress != nil:
					progress++
				default:
					tokens++
					if len(chunk.CompletionProbabilities) == 0 || chunk.CompletionProbabilities[0].Prob == nil {
						t.Fatalf("missing post-sampling probabilities: %+v", chunk)
					}
				}
			}
			if progress == 0 || tokens == 0 || final != 1 {
				t.Fatalf("progress=%d tokens=%d final=%d", progress, tokens, final)
			}
			if ex := obs.Exchanges(); len(ex) != 1 || !ex[0].BodyClosed() {
				t.Fatalf("stream did not close its response: %+v", ex)
			}
		})
	}
}

func TestSlotsCachePersistence(t *testing.T) {
	c, _ := slotsPreset.server(t).Client(t)
	ctx := t.Context()
	req := &llama.CompletionRequest{Prompt: llama.TextPrompt("A family once lived in a little house near the woods."), IDSlot: new(0), NPredict: new(1), CachePrompt: new(true)}
	if _, err := c.EraseSlot(ctx, 0, nil); err != nil {
		t.Fatal(err)
	}
	cold, err := c.Completion(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	file := &llama.SlotFileRequest{Filename: "roundtrip.bin"}
	saved, err := c.SaveSlot(ctx, 0, file)
	if err != nil {
		t.Fatal(err)
	}
	if saved.NSaved == 0 || saved.NWritten == 0 {
		t.Fatalf("empty slot save: %+v", saved)
	}
	erased, err := c.EraseSlot(ctx, 0, nil, llama.WithQuery("action", "restore"))
	if err != nil || erased.NErased != saved.NSaved {
		t.Fatalf("erase after save: %+v, %v; saved %+v", erased, err, saved)
	}
	restored, err := c.RestoreSlot(ctx, 0, file)
	if err != nil || restored.NRead != saved.NWritten || restored.NRestored != saved.NSaved {
		t.Fatalf("restore after erase: %+v, %v; saved %+v", restored, err, saved)
	}
	warm, err := c.Completion(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if cold.Timings == nil || warm.Timings == nil || warm.Timings.CacheN <= cold.Timings.CacheN {
		t.Fatalf("restored prompt did not reuse cached tokens: cold=%+v warm=%+v", cold.Timings, warm.Timings)
	}
}

func TestSlotsBusyDoesNotRetry(t *testing.T) {
	c, obs := slotsPreset.server(t).Client(t)
	received := false
	for _, err := range c.CompletionStream(t.Context(), &llama.CompletionRequest{Prompt: llama.TextPrompt("Once upon a time"), NPredict: new(512), IgnoreEOS: new(true)}) {
		if err != nil {
			t.Fatal(err)
		}
		received = true
		obs.Reset()
		_, err = c.Slots(t.Context(), llama.WithFailOnNoSlot())
		e, ok := errors.AsType[*llama.APIError](err)
		if !ok || e.StatusCode != http.StatusServiceUnavailable || len(obs.Exchanges()) != 1 {
			t.Fatalf("busy slots should return one HTTP 503, got %v; exchanges %+v", err, obs.Exchanges())
		}
		break
	}
	if !received {
		t.Fatal("completion stream yielded no event")
	}
}

func TestCompletionMediaRefusal(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	_, err := c.Completion(t.Context(), &llama.CompletionRequest{Prompt: llama.MultimodalPrompt("<__media__>", []byte{1}), NPredict: new(1)})
	e, ok := errors.AsType[*llama.APIError](err)
	if !ok || e.Type != llama.ErrorTypeInvalidRequest {
		t.Fatalf("media on a text-only model: %v", err)
	}
}

func TestSlotsNegativeIDNoRequest(t *testing.T) {
	c, obs := slotsPreset.server(t).Client(t)
	ctx := t.Context()

	tests := map[string]struct {
		call func() error
	}{
		"error: SaveSlot negative id": {
			call: func() error {
				_, err := c.SaveSlot(ctx, -1, &llama.SlotFileRequest{Filename: "negative.bin"})
				return err
			},
		},
		"error: RestoreSlot negative id": {
			call: func() error {
				_, err := c.RestoreSlot(ctx, -1, &llama.SlotFileRequest{Filename: "negative.bin"})
				return err
			},
		},
		"error: EraseSlot negative id": {
			call: func() error {
				_, err := c.EraseSlot(ctx, -1, nil)
				return err
			},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			obs.Reset()
			err := tt.call()
			if err == nil {
				t.Fatal("expected error for negative slot id")
			}
			if !strings.Contains(err.Error(), "non-negative") {
				t.Errorf("error %q does not explain that slot id must be non-negative", err)
			}
			if ex := obs.Exchanges(); len(ex) != 0 {
				t.Fatalf("negative slot id sent %d HTTP request(s): %+v", len(ex), ex)
			}
		})
	}
}

func TestEraseSlotIDRange(t *testing.T) {
	tests := map[string]struct {
		id     string
		status int
	}{
		"success: signed 32-bit maximum accepted": {"2147483647", http.StatusOK},
		"error: above signed 32-bit maximum":      {"2147483648", http.StatusBadRequest},
	}
	for name, tt := range tests {
		if strconv.IntSize == 32 && tt.status == http.StatusBadRequest {
			// This ID cannot be supplied through the typed API on 32-bit systems.
			continue
		}
		t.Run(name, func(t *testing.T) {
			id, err := strconv.Atoi(tt.id)
			if err != nil {
				t.Fatal(err)
			}
			c, obs := slotsPreset.server(t).Client(t)
			out, err := c.EraseSlot(t.Context(), id, nil)
			if tt.status == http.StatusOK {
				if err != nil || out == nil || out.IDSlot != id {
					t.Errorf("EraseSlot(%d) = %+v, %v; want the requested ID in a successful response", id, out, err)
				}
			} else if e, ok := errors.AsType[*llama.APIError](err); !ok || e.StatusCode != tt.status || e.Type != llama.ErrorTypeInvalidRequest || e.Message != "Invalid slot ID" {
				t.Errorf("EraseSlot(%d) error = %v; want HTTP 400 invalid_request_error: Invalid slot ID", id, err)
			}
			if ex := obs.Exchanges(); len(ex) != 1 || ex[0].StatusCode != tt.status {
				t.Errorf("EraseSlot(%d) exchanges = %+v; want one HTTP %d", id, ex, tt.status)
			}
		})
	}
}

func TestPropsUISettingsVariants(t *testing.T) {
	model := harness.Model(t, "stories260K")
	tests := map[string]struct {
		uiConfig string
	}{
		"success: array":  {uiConfig: `[]`},
		"success: null":   {uiConfig: `null`},
		"success: object": {uiConfig: `{"theme":"dark"}`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			srv := harness.StartT(t, harness.Options{
				Model: model,
				Args:  []string{"--ctx-size", "256", "--seed", "42", "--ui-config", tt.uiConfig},
			})
			c, _ := srv.Client(t)
			props, err := c.Props(t.Context())
			if err != nil {
				t.Fatalf("Props with --ui-config %s: %v", tt.uiConfig, err)
			}
			rawUI, err := json.Marshal(props.UISettings)
			if err != nil {
				t.Fatalf("encoding ui_settings: %v", err)
			}
			got := jsontext.Value(rawUI)
			if err := got.Canonicalize(); err != nil {
				t.Fatalf("canonicalizing ui_settings %q: %v", got, err)
			}
			want := jsontext.Value(tt.uiConfig)
			if err := want.Canonicalize(); err != nil {
				t.Fatalf("canonicalizing want %q: %v", want, err)
			}
			if string(got) != string(want) {
				t.Errorf("UISettings = %s, want %s", got, want)
			}
		})
	}
}

func TestPropsVoices(t *testing.T) {
	c, _ := presetRealtimeOmni.server(t).Client(t)
	props, err := c.Props(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"female", "male"}; !slices.Equal(props.Voices, want) {
		t.Errorf("Voices = %v, want %v", props.Voices, want)
	}
}
