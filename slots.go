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
	"encoding/json/jsontext"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
)

// Slot reports the state of one inference slot. Task-specific members are absent until the slot has run a task.
type Slot struct {
	ID                     int                       `json:"id"`
	NCtx                   int                       `json:"n_ctx"`
	Speculative            bool                      `json:"speculative"`
	IsProcessing           bool                      `json:"is_processing"`
	IDTask                 *int                      `json:"id_task,omitzero"`
	NPromptTokens          *int                      `json:"n_prompt_tokens,omitzero"`
	NPromptTokensProcessed *int                      `json:"n_prompt_tokens_processed,omitzero"`
	NPromptTokensCache     *int                      `json:"n_prompt_tokens_cache,omitzero"`
	Params                 *GenerationSettings       `json:"params,omitzero"`
	NextToken              []SlotNextToken           `json:"next_token,omitzero"`
	Prompt                 *string                   `json:"prompt,omitzero"`
	Generated              *string                   `json:"generated,omitzero"`
	Extra                  map[string]jsontext.Value `json:",embed"`
}

// SlotNextToken describes token generation for a slot that has processed a task.
type SlotNextToken struct {
	HasNextToken bool                      `json:"has_next_token"`
	HasNewLine   bool                      `json:"has_new_line"`
	NRemain      int                       `json:"n_remain"`
	NDecoded     int                       `json:"n_decoded"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// SlotFileRequest names a file in the server's slot save directory, not the client's filesystem.
type SlotFileRequest struct {
	Filename string                    `json:"filename"`
	Model    *string                   `json:"model,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// SlotEraseRequest optionally selects the model on a router; a single-model server needs no body.
type SlotEraseRequest struct {
	Model *string                   `json:"model,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotSaveResponse reports how many cached tokens and bytes were written.
type SlotSaveResponse struct {
	IDSlot   int                       `json:"id_slot"`
	Filename string                    `json:"filename"`
	NSaved   int                       `json:"n_saved"`
	NWritten int64                     `json:"n_written"`
	Timings  SlotSaveTimings           `json:"timings"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// SlotSaveTimings measures the time taken to save a slot, in milliseconds.
type SlotSaveTimings struct {
	SaveMS float64                   `json:"save_ms"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// SlotRestoreResponse reports how many cached tokens and bytes were restored.
type SlotRestoreResponse struct {
	IDSlot    int                       `json:"id_slot"`
	Filename  string                    `json:"filename"`
	NRestored int                       `json:"n_restored"`
	NRead     int64                     `json:"n_read"`
	Timings   SlotRestoreTimings        `json:"timings"`
	Extra     map[string]jsontext.Value `json:",embed"`
}

// SlotRestoreTimings measures the time taken to restore a slot, in milliseconds.
type SlotRestoreTimings struct {
	RestoreMS float64                   `json:"restore_ms"`
	Extra     map[string]jsontext.Value `json:",embed"`
}

// SlotEraseResponse reports how many tokens the server erased from the slot.
type SlotEraseResponse struct {
	IDSlot  int                       `json:"id_slot"`
	NErased int                       `json:"n_erased"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// WithFailOnNoSlot makes Slots return 503 immediately, without a retry, when no slot is available.
func WithFailOnNoSlot() RequestOption {
	return func(rc *requestConfig) { WithQuery("fail_on_no_slot", "1")(rc); WithNoRetry()(rc) }
}

// Slots reports the state of every inference slot.
func (c *Client) Slots(ctx context.Context, opts ...RequestOption) ([]Slot, error) {
	return Call[[]Slot](ctx, c, http.MethodGet, "/slots", nil, opts...)
}

// SaveSlot saves the prompt cache of a slot to req.Filename on the server.
//
// Negative slot ids are rejected before sending a request because the server converts negative ids to unsigned and
// wraps modulo the slot count. An out-of-range non-negative id wraps around the server's slot count; the client
// cannot know the count.
func (c *Client) SaveSlot(ctx context.Context, id int, req *SlotFileRequest, opts ...RequestOption) (*SlotSaveResponse, error) {
	return slotsAction[*SlotSaveResponse](ctx, c, id, "save", req, opts)
}

// RestoreSlot restores a slot's prompt cache from req.Filename on the server.
//
// Negative slot ids are rejected before sending a request because the server converts negative ids to unsigned and
// wraps modulo the slot count. An out-of-range non-negative id wraps around the server's slot count; the client
// cannot know the count.
func (c *Client) RestoreSlot(ctx context.Context, id int, req *SlotFileRequest, opts ...RequestOption) (*SlotRestoreResponse, error) {
	return slotsAction[*SlotRestoreResponse](ctx, c, id, "restore", req, opts)
}

// EraseSlot removes the cached tokens of a slot. Pass nil for a single-model server.
//
// Negative slot ids are rejected before sending a request because the server converts negative ids to unsigned and
// wraps modulo the slot count. An out-of-range non-negative id wraps around the server's slot count; the client
// cannot know the count.
func (c *Client) EraseSlot(ctx context.Context, id int, req *SlotEraseRequest, opts ...RequestOption) (*SlotEraseResponse, error) {
	if req == nil {
		return slotsAction[*SlotEraseResponse](ctx, c, id, "erase", nil, opts)
	}
	return slotsAction[*SlotEraseResponse](ctx, c, id, "erase", req, opts)
}

func slotsAction[T any](ctx context.Context, c *Client, id int, action string, body any, opts []RequestOption) (T, error) {
	if id < 0 {
		var zero T
		return zero, fmt.Errorf("llama: negative slot id %d (slot ids must be non-negative; the server converts negative ids to unsigned and wraps modulo the slot count)", id)
	}
	// A caller-supplied generic query option must not change which operation a typed method performs.
	opts = append(slices.Clip(opts), func(rc *requestConfig) {
		if rc.query == nil {
			rc.query = url.Values{}
		}
		rc.query.Set("action", action)
	})
	return Call[T](ctx, c, http.MethodPost, "/slots/"+strconv.Itoa(id), body, opts...)
}
