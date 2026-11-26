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

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"slices"
)

// Message is a user or assistant turn in the conversation history.
type Message struct {
	// Role identifies who supplied the turn, typically "user" or "assistant".
	Role string `json:"role"`
	// Content is text or a list of blocks; the server skips an assistant turn with omitted content.
	Content Content `json:"content,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Content holds a string, a list of blocks, an explicit null, or any other JSON value the server
// tolerates. Its zero value omits an optional member.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type Content struct {
	text   *string
	blocks []ContentBlock
	null   bool
	raw    jsontext.Value
}

// TextContent returns content encoded as one string, including an empty one.
func TextContent(s string) Content { return Content{text: &s} }

// BlocksContent returns content encoded as an array, including an empty one.
func BlocksContent(blocks ...ContentBlock) Content {
	if blocks == nil {
		blocks = []ContentBlock{}
	}
	return Content{blocks: slices.Clone(blocks)}
}

// NullContent returns an explicit null instead of omitting the member.
func NullContent() Content { return Content{null: true} }

// RawContent returns content that encodes v without interpreting it. The server converts a tool
// result's non-string, non-array content to an empty string.
func RawContent(v jsontext.Value) Content { return Content{raw: slices.Clone(v)} }

// Text returns the string and whether the content was given in string form.
func (c Content) Text() (string, bool) {
	if c.text == nil {
		return "", false
	}
	return *c.text, true
}

// Blocks returns a read-only view of the typed array of blocks, or nil when content is not a block array.
func (c Content) Blocks() []ContentBlock { return c.blocks }

// IsNull reports whether the content is explicitly null.
func (c Content) IsNull() bool { return c.null }

// Raw returns a read-only view of the stored JSON when the content uses the raw form, or nil otherwise.
func (c Content) Raw() jsontext.Value { return c.raw }

// IsZero reports whether the content is unset.
func (c Content) IsZero() bool { return c.text == nil && c.blocks == nil && !c.null && c.raw == nil }

// MarshalJSONTo encodes the content in its original wire form.
func (c Content) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case c.text != nil:
		return enc.WriteToken(jsontext.String(*c.text))
	case c.blocks != nil:
		return json.MarshalEncode(enc, c.blocks)
	case c.raw != nil:
		return enc.WriteValue(c.raw)
	default:
		return enc.WriteToken(jsontext.Null)
	}
}

// UnmarshalJSONFrom decodes a string, an array of blocks, or null. It preserves other valid
// JSON as raw content, including arrays that cannot be represented as typed blocks.
func (c *Content) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch dec.PeekKind() {
	case '"':
		var s string
		if err := json.UnmarshalDecode(dec, &s); err != nil {
			return err
		}
		*c = TextContent(s)
	case '[':
		raw, err := dec.ReadValue()
		if err != nil {
			return err
		}
		var blocks []ContentBlock
		if err := json.Unmarshal(raw, &blocks, dec.Options()); err != nil {
			if strict, _ := json.GetOption(dec.Options(), json.RejectUnknownMembers); strict {
				return err
			}
			*c = RawContent(raw)
			return nil
		}
		*c = Content{blocks: blocks}
	case 'n':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*c = NullContent()
	default:
		raw, err := dec.ReadValue()
		if err != nil {
			return err
		}
		*c = RawContent(raw)
	}
	return nil
}

// ContentBlock is one block of message content. System instructions use only text blocks.
type ContentBlock struct {
	// Type selects text, thinking, image, tool_use, tool_result or a nested tool_reference; unknown or absent types
	// are skipped.
	Type string `json:"type,omitzero"`
	// Text supplies a text block's payload; system and tool-result conversions treat omission as an empty string.
	Text *string `json:"text,omitzero"`
	// Thinking is model reasoning, concatenated into the reasoning history when replayed.
	Thinking *string `json:"thinking,omitzero"`
	// Signature is empty in thinking blocks emitted by local models.
	Signature *string `json:"signature,omitzero"`
	// Source specifies the bytes or URL for an image block; the converter skips an unrecognized source type.
	Source *ImageSource `json:"source,omitzero"`
	// ID links a tool_use block to a later tool_result with the same ToolUseID.
	ID *string `json:"id,omitzero"`
	// Name is the called function's name in a tool_use block.
	Name *string `json:"name,omitzero"`
	// Input holds tool arguments as a JSON value, not a JSON-encoded string; omitted tool_use input defaults to {}.
	Input jsontext.Value `json:"input,omitzero"`
	// ToolUseID identifies the call a tool_result block answers.
	ToolUseID *string `json:"tool_use_id,omitzero"`
	// Content is a tool result's text or nested blocks; an omitted value becomes an empty string.
	Content Content `json:"content,omitzero"`
	// ToolName identifies a deferred tool to expose through a nested tool_reference.
	ToolName *string `json:"tool_name,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
	null  bool
}

// IsNull reports whether the block represents a null array element ignored by the server.
func (b ContentBlock) IsNull() bool { return b.null }

// MarshalJSONTo writes the block or its explicit null value.
func (b ContentBlock) MarshalJSONTo(enc *jsontext.Encoder) error {
	if b.null {
		return enc.WriteToken(jsontext.Null)
	}
	type wire ContentBlock
	return json.MarshalEncode(enc, wire(b))
}

// UnmarshalJSONFrom reads a block and preserves an explicit null instead of replacing it with an object.
func (b *ContentBlock) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if dec.PeekKind() == 'n' {
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*b = ContentBlock{null: true}
		return nil
	}
	type wire ContentBlock
	if err := json.UnmarshalDecode(dec, (*wire)(b)); err != nil {
		return err
	}
	b.null = false
	return nil
}

// ImageSource supplies an image by URL or as base64 data.
type ImageSource struct {
	// Type selects "base64" or "url"; the converter does not interpret an omitted type.
	Type string `json:"type"`
	// MediaType is the MIME type of base64 data; the server defaults to image/jpeg.
	MediaType *string `json:"media_type,omitzero"`
	// Data contains base64-encoded image bytes without a data URL prefix for a "base64" source.
	Data *string `json:"data,omitzero"`
	// URL is fetched or decoded by the server for a "url" source, subject to its media access settings.
	URL *string `json:"url,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}
