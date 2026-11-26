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

package anthropic

import "encoding/json/jsontext"

// MessageEvent is the data of a named event in a MessagesStream.
type MessageEvent struct {
	// Type is message_start, content_block_start, content_block_delta, content_block_stop, message_delta or message_stop.
	Type string `json:"type"`
	// Message is the initial assistant message in message_start.
	Message *MessageResponse `json:"message,omitzero"`
	// Index identifies the content block being opened, updated or closed.
	Index *int `json:"index,omitzero"`
	// ContentBlock is the initial block in content_block_start.
	ContentBlock *ContentBlock `json:"content_block,omitzero"`
	// Delta carries the update in content_block_delta and message_delta.
	Delta *Delta `json:"delta,omitzero"`
	// Usage is the output token count in message_delta.
	Usage *DeltaUsage               `json:"usage,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// Delta is a content block fragment or the final stop information of an assistant message.
type Delta struct {
	// Type is text_delta, thinking_delta, signature_delta or input_json_delta; message_delta has no type.
	Type *string `json:"type,omitzero"`
	// Text appends text to a block.
	Text *string `json:"text,omitzero"`
	// Thinking appends reasoning to a block.
	Thinking *string `json:"thinking,omitzero"`
	// Signature is empty when llama.cpp finishes a thinking block.
	Signature *string `json:"signature,omitzero"`
	// PartialJSON appends a fragment of the tool arguments, not necessarily complete JSON by itself.
	PartialJSON *string `json:"partial_json,omitzero"`
	// StopDetails is present only in message_delta.
	*StopDetails
	Extra map[string]jsontext.Value `json:",embed"`
}

// StopDetails is the final message_delta payload. Unlike a content delta it always has a stop_sequence member, even when null.
type StopDetails struct {
	// StopReason is "end_turn", "max_tokens" or "tool_use".
	StopReason string `json:"stop_reason"`
	// StopSequence holds the matched stop string, or null when none was matched.
	StopSequence *string `json:"stop_sequence"`
}

// DeltaUsage is the generation count in message_delta, which does not repeat the prompt counts.
type DeltaUsage struct {
	// OutputTokens is the total number of generated tokens.
	OutputTokens int                       `json:"output_tokens"`
	Extra        map[string]jsontext.Value `json:",embed"`
}
