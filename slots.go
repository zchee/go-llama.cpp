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
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
)

// Slot reports the state of one inference slot. Task-specific members are absent until the slot has run a task.
type Slot struct {
	// ID is the zero-based slot number used in save, restore, and erase requests.
	ID int `json:"id"`
	// NCtx is the context capacity of this slot, in tokens.
	NCtx int `json:"n_ctx"`
	// Speculative reports whether this slot can use speculative decoding.
	Speculative bool `json:"speculative"`
	// IsProcessing reports whether the slot currently has work rather than being idle.
	IsProcessing bool `json:"is_processing"`
	// IDTask identifies the current or most recent task; absent on a fresh slot.
	IDTask *int `json:"id_task,omitzero"`
	// NPromptTokens counts positions currently held in the slot's prompt cache; absent on a fresh slot.
	NPromptTokens *int `json:"n_prompt_tokens,omitzero"`
	// NPromptTokensProcessed counts input tokens evaluated for the task rather than reused; absent on a fresh slot.
	NPromptTokensProcessed *int `json:"n_prompt_tokens_processed,omitzero"`
	// NPromptTokensCache counts input tokens reused for the task; absent on a fresh slot.
	NPromptTokensCache *int `json:"n_prompt_tokens_cache,omitzero"`
	// Params reports the effective generation settings of the current or most recent task; absent on a fresh slot.
	Params *GenerationSettings `json:"params,omitzero"`
	// NextToken carries one generation state object for the current or most recent task; absent on a fresh slot.
	NextToken []SlotNextToken `json:"next_token,omitzero"`
	// Prompt is the detokenized input of the current or most recent task; omitted on a fresh slot.
	Prompt *string `json:"prompt,omitzero"`
	// Generated is the output text so far for the current or most recent task; omitted on a fresh slot.
	Generated *string `json:"generated,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotNextToken describes token generation for a slot that has processed a task.
type SlotNextToken struct {
	// HasNextToken reports whether generation can continue past the most recently produced token.
	HasNextToken bool `json:"has_next_token"`
	// HasNewLine reports whether a newline has been encountered during generation.
	HasNewLine bool `json:"has_new_line"`
	// NRemain is the remaining generation token budget, or -1 if unlimited.
	NRemain int `json:"n_remain"`
	// NDecoded counts tokens generated so far for the task.
	NDecoded int `json:"n_decoded"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotFileRequest names a file in the server's slot save directory, not the client's filesystem.
type SlotFileRequest struct {
	// Filename is the required cache file name under the server's --slot-save-path directory.
	Filename string `json:"filename"`
	// Model selects the model that owns the slot when routing; not needed on a single-model server.
	Model string `json:"model,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotEraseRequest optionally selects the model on a router; a single-model server needs no body.
type SlotEraseRequest struct {
	// Model identifies the model whose slot cache to clear when routing; not needed on a single-model server.
	Model string `json:"model,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotSaveResponse reports how many cached tokens and bytes were written.
type SlotSaveResponse struct {
	// IDSlot echoes the requested slot number, even when the server wraps it modulo the slot count.
	IDSlot int `json:"id_slot"`
	// Filename echoes the cache file name from the save request.
	Filename string `json:"filename"`
	// NSaved counts tokens serialized from the slot's cache.
	NSaved int `json:"n_saved"`
	// NWritten counts bytes written to the cache file.
	NWritten int64 `json:"n_written"`
	// Timings measures the duration of the server-side save.
	Timings SlotSaveTimings `json:"timings"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotSaveTimings measures the time taken to save a slot, in milliseconds.
type SlotSaveTimings struct {
	// SaveMS is the elapsed time in milliseconds to write the cache on the server.
	SaveMS float64 `json:"save_ms"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotRestoreResponse reports how many cached tokens and bytes were restored.
type SlotRestoreResponse struct {
	// IDSlot echoes the requested slot number, even when the server wraps it modulo the slot count.
	IDSlot int `json:"id_slot"`
	// Filename echoes the cache file name from the restore request.
	Filename string `json:"filename"`
	// NRestored counts tokens loaded into the slot's cache.
	NRestored int `json:"n_restored"`
	// NRead counts bytes read from the cache file.
	NRead int64 `json:"n_read"`
	// Timings measures the duration of the server-side restore.
	Timings SlotRestoreTimings `json:"timings"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotRestoreTimings measures the time taken to restore a slot, in milliseconds.
type SlotRestoreTimings struct {
	// RestoreMS is the elapsed time in milliseconds to load the saved cache on the server.
	RestoreMS float64 `json:"restore_ms"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SlotEraseResponse reports how many tokens the server erased from the slot.
type SlotEraseResponse struct {
	// IDSlot echoes the requested slot number, even when the server wraps it modulo the slot count.
	IDSlot int `json:"id_slot"`
	// NErased counts the cached tokens removed from the slot.
	NErased int `json:"n_erased"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// WithFailOnNoSlot asks Slots to return 503 when no slot is free.
// Unlike setting fail_on_no_slot in the path or with WithQuery, this option limits the library to one attempt even if the connection fails.
func WithFailOnNoSlot() RequestOption {
	return func(rc *requestConfig) {
		WithQuery("fail_on_no_slot", "1")(rc)
		rc.noRetry = true
	}
}

// Slots reports the state of every inference slot.
func (c *Client) Slots(ctx context.Context, opts ...RequestOption) ([]Slot, error) {
	return Call[[]Slot](ctx, c, http.MethodGet, "/slots", nil, opts...)
}

// SaveSlot saves the prompt cache of a slot to req.Filename on the server.
//
// Negative slot IDs are rejected before sending because the server converts them to unsigned and wraps
// modulo the slot count. A non-negative ID that fits in a C++ int also wraps modulo the slot count, which
// the client does not know. The server returns HTTP 400 for a larger ID.
func (c *Client) SaveSlot(ctx context.Context, id int, req *SlotFileRequest, opts ...RequestOption) (*SlotSaveResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil save slot request")
	}
	return slotsAction[*SlotSaveResponse](ctx, c, id, "save", req, opts)
}

// RestoreSlot restores a slot's prompt cache from req.Filename on the server.
//
// Negative slot IDs are rejected before sending because the server converts them to unsigned and wraps
// modulo the slot count. A non-negative ID that fits in a C++ int also wraps modulo the slot count, which
// the client does not know. The server returns HTTP 400 for a larger ID.
func (c *Client) RestoreSlot(ctx context.Context, id int, req *SlotFileRequest, opts ...RequestOption) (*SlotRestoreResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil restore slot request")
	}
	return slotsAction[*SlotRestoreResponse](ctx, c, id, "restore", req, opts)
}

// EraseSlot removes the cached tokens of a slot. Pass nil for a single-model server.
//
// Negative slot IDs are rejected before sending because the server converts them to unsigned and wraps
// modulo the slot count. A non-negative ID that fits in a C++ int also wraps modulo the slot count, which
// the client does not know. The server returns HTTP 400 for a larger ID.
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
