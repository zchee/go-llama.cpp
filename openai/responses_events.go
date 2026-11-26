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
type ResponseEvent struct {
	// Value holds the concrete JSON payload selected by the event's type discriminator.
	Value ResponseEventValue
}

// ResponseEventValue is a lifecycle, output item, content part, delta, text done or unknown event.
type ResponseEventValue interface{ responsesEventType() string }

// ResponseEventHeader carries the discriminator and optional llama.cpp streaming statistics.
type ResponseEventHeader struct {
	// Type matches the SSE event name and selects the JSON payload shape.
	Type string `json:"type"`
	// Timings reports prompt processing and generation statistics when the server has timings for the event.
	Timings *llama.Timings `json:"timings,omitzero"`
	// PromptProgress reports prompt-processing progress on updates enabled by SamplingParams.ReturnProgress.
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
	// Response carries only identity and status before completion, then the full output and token usage.
	Response Response `json:"response"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseItemEvent carries the item added to or finished in an output slot.
type ResponseItemEvent struct {
	ResponseEventHeader
	// OutputIndex is the zero-based slot of the item in the response output.
	OutputIndex int `json:"output_index"`
	// Item carries the initial or completed snapshot of the item at OutputIndex.
	Item ResponseItem `json:"item,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponsePartEvent carries the content part added to or finished in a message.
type ResponsePartEvent struct {
	ResponseEventHeader
	// ItemID identifies the message that owns the changed content part.
	ItemID string `json:"item_id"`
	// OutputIndex is the zero-based slot of the owning message in the response output.
	OutputIndex int `json:"output_index"`
	// ContentIndex is the zero-based position of the part within the owning message.
	ContentIndex int `json:"content_index"`
	// Part carries the output-text part, initially empty when added and containing the full text when done.
	Part ResponsePart `json:"part,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseDeltaEvent appends output text, reasoning text or function call arguments to an item.
// ContentIndex is absent on function call argument deltas.
type ResponseDeltaEvent struct {
	ResponseEventHeader
	// ItemID matches the ID of the item to which Delta belongs, not the function's CallID.
	ItemID string `json:"item_id"`
	// OutputIndex is the zero-based slot of the updated item in the response output.
	OutputIndex int `json:"output_index"`
	// ContentIndex is the zero-based position of the text part, or nil for function arguments.
	ContentIndex *int `json:"content_index,omitzero"`
	// Delta is the next text fragment or a fragment of the JSON argument string, not necessarily valid JSON by itself.
	Delta string `json:"delta"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseTextDoneEvent carries the entire text of a completed output text part.
type ResponseTextDoneEvent struct {
	ResponseEventHeader
	// ItemID identifies the assistant message whose output text part finished.
	ItemID string `json:"item_id"`
	// OutputIndex is the zero-based slot of the message in the response output.
	OutputIndex int `json:"output_index"`
	// ContentIndex is the zero-based position of the finished part within the message.
	ContentIndex int `json:"content_index"`
	// Text is the full finished text, not a fragment to append to earlier deltas.
	Text string `json:"text"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
