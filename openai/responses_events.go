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

package openai

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"

	llama "github.com/zchee/go-llama.cpp"
)

// ResponseEvent is a named SSE payload. Switch on Value for the fields of each event family.
type ResponseEvent struct{ Value ResponseEventValue }

// ResponseEventValue is a lifecycle, output item, content part, delta, text done or unknown event.
type ResponseEventValue interface{ responsesEventType() string }

// ResponseEventHeader carries the discriminator and optional llama.cpp streaming statistics.
type ResponseEventHeader struct {
	Type           string                `json:"type"`
	Timings        *llama.Timings        `json:"timings,omitzero"`
	PromptProgress *llama.PromptProgress `json:"prompt_progress,omitzero"`
}

func (h *ResponseEventHeader) responsesEventType() string { return h.Type }
func (u *ResponseUnknown) responsesEventType() string     { return u.Type }

// ResponseType returns the SSE event name, which llama.cpp also includes in the JSON payload.
func (e ResponseEvent) ResponseType() string {
	if e.Value == nil {
		return ""
	}
	return e.Value.responsesEventType()
}

// ResponseLifecycleEvent carries the response.created, response.in_progress or response.completed snapshot.
type ResponseLifecycleEvent struct {
	ResponseEventHeader
	Response Response                  `json:"response"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// ResponseItemEvent carries the item added to or finished in an output slot.
type ResponseItemEvent struct {
	ResponseEventHeader
	OutputIndex int                       `json:"output_index"`
	Item        ResponseItem              `json:"item,omitzero"`
	Extra       map[string]jsontext.Value `json:",embed"`
}

// ResponsePartEvent carries the content part added to or finished in a message.
type ResponsePartEvent struct {
	ResponseEventHeader
	ItemID       string                    `json:"item_id"`
	OutputIndex  int                       `json:"output_index"`
	ContentIndex int                       `json:"content_index"`
	Part         ResponsePart              `json:"part,omitzero"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// ResponseDeltaEvent appends output text, reasoning text or function call arguments to an item.
// ContentIndex is absent on function call argument deltas.
type ResponseDeltaEvent struct {
	ResponseEventHeader
	ItemID       string                    `json:"item_id"`
	OutputIndex  int                       `json:"output_index"`
	ContentIndex *int                      `json:"content_index,omitzero"`
	Delta        string                    `json:"delta"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// ResponseTextDoneEvent carries the entire text of a completed output text part.
type ResponseTextDoneEvent struct {
	ResponseEventHeader
	ItemID       string                    `json:"item_id"`
	OutputIndex  int                       `json:"output_index"`
	ContentIndex int                       `json:"content_index"`
	Text         string                    `json:"text"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// MarshalJSONTo writes the selected event's JSON payload, not its SSE frame.
func (e ResponseEvent) MarshalJSONTo(enc *jsontext.Encoder) error {
	return json.MarshalEncode(enc, e.Value)
}

// UnmarshalJSONFrom selects the event family by type and keeps unknown events intact.
func (e *ResponseEvent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return responsesReadTagged(dec, &e.Value, func(kind string) ResponseEventValue {
		switch kind {
		case "response.created", "response.in_progress", "response.completed":
			return new(ResponseLifecycleEvent)
		case "response.output_item.added", "response.output_item.done":
			return new(ResponseItemEvent)
		case "response.content_part.added", "response.content_part.done":
			return new(ResponsePartEvent)
		case "response.output_text.delta", "response.reasoning_text.delta", "response.function_call_arguments.delta":
			return new(ResponseDeltaEvent)
		case "response.output_text.done":
			return new(ResponseTextDoneEvent)
		default:
			return new(ResponseUnknown)
		}
	})
}
