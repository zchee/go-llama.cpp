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
	"fmt"
	"net/url"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
)

var _ = registerContracts(map[string]contractCase{
	"POST /completion": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		r, err := c.Completion(t.Context(), &llama.CompletionRequest{Prompt: llama.TextPrompt("Once"), NPredict: new(1)})
		if err == nil && (r.Stop == nil || !*r.Stop) {
			return fmt.Errorf("missing final completion: %+v", r)
		}
		return err
	}},
	"POST /infill": {server: infillPreset, call: func(t *testing.T, c *llama.Client) error {
		r, err := c.Infill(t.Context(), &llama.InfillRequest{InputPrefix: llama.InfillInput{Text: "Once"}, InputSuffix: llama.InfillInput{Text: "end"}, NPredict: new(1)})
		if err == nil && (r.Stop == nil || !*r.Stop) {
			return fmt.Errorf("missing final infill: %+v", r)
		}
		return err
	}},
	"GET /props": {server: slotsPreset, call: func(t *testing.T, c *llama.Client) error {
		r, err := c.Props(t.Context())
		if err == nil && r.DefaultGenerationSettings.Params == nil {
			return fmt.Errorf("model props lack defaults: %+v", r)
		}
		return err
	}},
	"POST /props": {server: slotsPreset, call: func(t *testing.T, c *llama.Client) error {
		r, err := c.SetProps(t.Context(), &llama.PropsSetRequest{})
		if err == nil && !r.Success {
			return fmt.Errorf("props not acknowledged: %+v", r)
		}
		return err
	}},
	"GET /slots": {server: slotsPreset, query: url.Values{"fail_on_no_slot": {"1"}}, call: func(t *testing.T, c *llama.Client) error {
		r, err := c.Slots(t.Context(), llama.WithFailOnNoSlot())
		if err == nil && len(r) != 1 {
			return fmt.Errorf("got %d slots, want 1", len(r))
		}
		return err
	}},
	"POST /slots/{id_slot}?action=save": {server: slotsPreset, call: func(t *testing.T, c *llama.Client) error {
		_, err := c.SaveSlot(t.Context(), 0, &llama.SlotFileRequest{Filename: "save-contract.bin"})
		return err
	}},
	"POST /slots/{id_slot}?action=restore": {server: slotsPreset, call: func(t *testing.T, c *llama.Client) error {
		file := &llama.SlotFileRequest{Filename: "restore-contract.bin"}
		if _, err := c.Completion(t.Context(), &llama.CompletionRequest{Prompt: llama.TextPrompt("Once"), NPredict: new(1), IDSlot: new(0)}); err != nil {
			return err
		}
		if _, err := c.SaveSlot(t.Context(), 0, file); err != nil {
			return err
		}
		_, err := c.RestoreSlot(t.Context(), 0, file)
		return err
	}},
	"POST /slots/{id_slot}?action=erase": {server: slotsPreset, call: func(t *testing.T, c *llama.Client) error { _, err := c.EraseSlot(t.Context(), 0, nil); return err }},
})
