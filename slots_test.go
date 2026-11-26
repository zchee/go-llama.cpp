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
	"log/slog"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestSlotsDebugFixture(t *testing.T) {
	data := jsontest.ReadFile(t, "testdata/slots/used_debug.resp.json")
	got := jsontest.RoundTrip[[]Slot](t, data)
	jsontest.InjectUnknown[[]Slot](t, data)
	jsontest.NoUnknown[[]Slot](t, data, nil)
	jsontest.CheckFallbacks[[]Slot](t)

	slots := *got
	if len(slots) != 2 {
		t.Fatalf("got %d slots, want 2", len(slots))
	}
	used := slots[0]
	if used.Prompt == nil || *used.Prompt != "<s> Once upon a time" {
		t.Errorf("Prompt = %v, want %q", used.Prompt, "<s> Once upon a time")
	}
	if used.Generated == nil || *used.Generated != ", there was a" {
		t.Errorf("Generated = %v, want %q", used.Generated, ", there was a")
	}
	if used.Params == nil {
		t.Fatal("Params is nil")
	}
	if diff := gocmp.Diff([]string{"\n", ":", `"`, "*"}, used.Params.DrySequenceBreakers); diff != "" {
		t.Errorf("DrySequenceBreakers (-want +got):\n%s", diff)
	}
	if used.Params.Grammar == nil || used.Params.GrammarLazy == nil {
		t.Errorf("expected full grammar params from slots debug response: %+v", used.Params)
	}
}

func TestSlotsNegativeIDRejected(t *testing.T) {
	var logs bytes.Buffer
	c, err := New("http://"+closedAddr(t), WithLogger(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))))
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()

	tests := map[string]struct {
		id   int
		call func(context.Context, int) error
	}{
		"error: SaveSlot negative id": {
			id: -1,
			call: func(ctx context.Context, id int) error {
				_, err := c.SaveSlot(ctx, id, &SlotFileRequest{Filename: "slot.bin"})
				return err
			},
		},
		"error: RestoreSlot negative id": {
			id: -1,
			call: func(ctx context.Context, id int) error {
				_, err := c.RestoreSlot(ctx, id, &SlotFileRequest{Filename: "slot.bin"})
				return err
			},
		},
		"error: EraseSlot negative id": {
			id: -42,
			call: func(ctx context.Context, id int) error {
				_, err := c.EraseSlot(ctx, id, nil)
				return err
			},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			logs.Reset()
			err := tt.call(t.Context(), tt.id)
			if err == nil {
				t.Fatalf("expected error for negative slot id %d", tt.id)
			}
			if !strings.Contains(err.Error(), "non-negative") {
				t.Errorf("error %q does not explain that slot id must be non-negative", err)
			}
			if logs.Len() != 0 {
				t.Errorf("client logged request activity for rejected negative slot id:\n%s", logs.String())
			}
		})
	}
}
