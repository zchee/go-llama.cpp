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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"reflect"
	"slices"
)

// The chat message types of the OpenAI-compatible chat completions API. They live in this package because native
// endpoints take them too: /apply-template reads chat messages, and the multimodal input of /embeddings is a list
// of {content: [parts]} objects built from the chat content parts. The openai package re-exports them under the
// same names.

// Message is one message of a chat conversation, in requests and in responses. An empty Role, ContentPart.Type or
// ImageURL.URL is encoded as an absent member, which the server reads the same way.
type Message struct {
	// Role is "system", "developer", "user", "assistant" or "tool".
	Role string `json:"role,omitzero"`
	// Content is the message content. An assistant message that only calls tools may leave it unset.
	Content MessageContent `json:"content,omitzero"`
	// Name names the author of the message.
	Name *string `json:"name,omitzero"`
	// ReasoningContent is the reasoning of an assistant message, which llama.cpp returns separately from the
	// content for reasoning models.
	ReasoningContent *string `json:"reasoning_content,omitzero"`
	// ToolCalls are the tool calls of an assistant message.
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
	// ToolCallID is the id of the tool call a "tool" message answers.
	ToolCallID *string `json:"tool_call_id,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ContentKind is the JSON shape of a MessageContent.
type ContentKind int

const (
	// ContentKindNone is the zero MessageContent, omitted from requests.
	ContentKindNone ContentKind = iota
	// ContentKindText is a string.
	ContentKindText
	// ContentKindParts is an array of content parts.
	ContentKindParts
	// ContentKindNull is an explicit JSON null.
	ContentKindNull
)

// MessageContent is the "content" of a Message: a string, an array of content parts, or null. Build one with
// TextContent, PartsContent or NullContent.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type MessageContent struct {
	kind  ContentKind
	text  string
	parts []ContentPart
}

// TextContent returns content given as a string.
func TextContent(s string) MessageContent { return MessageContent{kind: ContentKindText, text: s} }

// PartsContent returns content given as content parts, such as text and images.
func PartsContent(parts ...ContentPart) MessageContent {
	if parts == nil {
		parts = []ContentPart{}
	}
	return MessageContent{kind: ContentKindParts, parts: slices.Clone(parts)}
}

// NullContent returns content that is an explicit JSON null.
func NullContent() MessageContent { return MessageContent{kind: ContentKindNull} }

// Kind returns the shape of the content.
func (m MessageContent) Kind() ContentKind { return m.kind }

// Text returns the string of ContentKindText content.
func (m MessageContent) Text() string { return m.text }

// Parts returns a read-only view of the parts of ContentKindParts content.
func (m MessageContent) Parts() []ContentPart { return m.parts }

// IsZero reports whether the content is unset.
func (m MessageContent) IsZero() bool { return m.kind == ContentKindNone }

// Equal reports whether two contents have the same shape and content.
func (m MessageContent) Equal(o MessageContent) bool {
	return m.kind == o.kind && m.text == o.text && reflect.DeepEqual(m.parts, o.parts)
}

// MarshalJSONTo implements json.MarshalerTo.
func (m MessageContent) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch m.kind {
	case ContentKindText:
		return enc.WriteToken(jsontext.String(m.text))
	case ContentKindParts:
		return json.MarshalEncode(enc, m.parts)
	default:
		return enc.WriteToken(jsontext.Null)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (m *MessageContent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '"':
		var s string
		if err := json.UnmarshalDecode(dec, &s); err != nil {
			return err
		}
		*m = TextContent(s)
	case '[':
		var parts []ContentPart
		if err := json.UnmarshalDecode(dec, &parts); err != nil {
			return err
		}
		*m = PartsContent(parts...)
	case 'n':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*m = NullContent()
	default:
		return fmt.Errorf("llama: content must be a string, an array or null, got %s", kind)
	}
	return nil
}

// ContentPart is one part of a MessageContent.
type ContentPart struct {
	// Type is "text", "image_url", "input_audio", "input_video" or "video_url".
	Type string `json:"type,omitzero"`
	// Text is the text of a "text" part.
	Text *string `json:"text,omitzero"`
	// ImageURL is the image of an "image_url" part.
	ImageURL *ImageURL `json:"image_url,omitzero"`
	// InputAudio is the audio of an "input_audio" part.
	InputAudio *MediaInput `json:"input_audio,omitzero"`
	// InputVideo is the video of an "input_video" part.
	InputVideo *MediaInput `json:"input_video,omitzero"`
	// VideoURL is the video of a "video_url" part.
	VideoURL *MediaInput `json:"video_url,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// TextContentPart returns a "text" content part.
func TextContentPart(s string) ContentPart { return ContentPart{Type: "text", Text: &s} }

// ImageURLContentPart returns an "image_url" content part. url is an http(s) URL, a data: URL with base64 data, or bare
// base64 data.
func ImageURLContentPart(url string) ContentPart {
	return ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: url}}
}

// ImageURL is the image of an "image_url" content part.
type ImageURL struct {
	// URL is an http(s) URL, a data: URL with base64 data, or bare base64 data.
	URL string `json:"url,omitzero"`
	// Detail is the OpenAI detail level; llama.cpp ignores it.
	Detail *string `json:"detail,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// MediaInput is the media of an "input_audio", "input_video" or "video_url" content part. The server reads Data
// and falls back to URL.
type MediaInput struct {
	// Data is base64 data or a data: URL.
	Data *string `json:"data,omitzero"`
	// URL is an http(s) URL or a data: URL.
	URL *string `json:"url,omitzero"`
	// Format is the media format, such as "wav"; llama.cpp does not need it.
	Format *string `json:"format,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolCall is a call of a tool by the model. In streamed deltas only Index and the parts that changed are set,
// so every member is optional.
type ToolCall struct {
	// Index is the position of the call, set in streamed deltas.
	Index *int `json:"index,omitzero"`
	// ID identifies the call; the "tool" message answering it repeats it.
	ID *string `json:"id,omitzero"`
	// Type is "function".
	Type *string `json:"type,omitzero"`
	// Function is the called function.
	Function *FunctionCall `json:"function,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionCall is the function of a ToolCall.
type FunctionCall struct {
	// Name is the function name.
	Name *string `json:"name,omitzero"`
	// Arguments is the JSON text of the arguments, or the next piece of it in a streamed delta.
	Arguments *string `json:"arguments,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}
