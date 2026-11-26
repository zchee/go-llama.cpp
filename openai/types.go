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
	"fmt"
	"reflect"

	llama "github.com/zchee/go-llama.cpp"
)

// Usage is the token usage of a chat or text completion.
type Usage = llama.Usage

// Message is one message of a chat conversation; see llama.Message.
type Message = llama.Message

// ContentKind is the JSON shape of a MessageContent; see llama.ContentKind.
type ContentKind = llama.ContentKind

// The shapes of a MessageContent.
const (
	ContentKindNone  = llama.ContentKindNone
	ContentKindText  = llama.ContentKindText
	ContentKindParts = llama.ContentKindParts
	ContentKindNull  = llama.ContentKindNull
)

// MessageContent is the "content" of a Message; see llama.MessageContent.
type MessageContent = llama.MessageContent

// ContentPart is one part of a MessageContent; see llama.ContentPart.
type ContentPart = llama.ContentPart

// ImageURL is the image of an "image_url" content part; see llama.ImageURL.
type ImageURL = llama.ImageURL

// MediaInput is the media of an "input_audio", "input_video" or "video_url" content part; see llama.MediaInput.
type MediaInput = llama.MediaInput

// ToolCall is a call of a tool by the model; see llama.ToolCall.
type ToolCall = llama.ToolCall

// FunctionCall is the function of a ToolCall; see llama.FunctionCall.
type FunctionCall = llama.FunctionCall

// TextContent returns content given as a string.
func TextContent(s string) MessageContent { return llama.TextContent(s) }

// PartsContent returns content given as content parts, such as text and images.
func PartsContent(parts ...ContentPart) MessageContent { return llama.PartsContent(parts...) }

// NullContent returns content that is an explicit JSON null.
func NullContent() MessageContent { return llama.NullContent() }

// TextPart returns a "text" content part.
func TextPart(s string) ContentPart { return llama.TextContentPart(s) }

// ImageURLPart returns an "image_url" content part. url is an http(s) URL, a data: URL with base64 data, or bare
// base64 data.
func ImageURLPart(url string) ContentPart { return llama.ImageURLContentPart(url) }

// InputTokensResponse is the response of POST /v1/chat/completions/input_tokens and POST
// /v1/responses/input_tokens: the number of tokens the input would take (handle_count_tokens in
// tools/server/server-context.cpp).
type InputTokensResponse struct {
	// Object is "response.input_tokens".
	Object string `json:"object"`
	// InputTokens is the number of input tokens.
	InputTokens int64 `json:"input_tokens"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// Tool is a tool the model may call. llama.cpp needs --jinja for tools.
type Tool struct {
	// Type is "function".
	Type string `json:"type"`
	// Function describes the function.
	Function *FunctionDefinition `json:"function,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionDefinition describes a function tool.
type FunctionDefinition struct {
	// Name is the function name.
	Name string `json:"name"`
	// Description tells the model what the function does.
	Description *string `json:"description,omitzero"`
	// Parameters is the JSON schema of the arguments; llama.SchemaFor builds one from a Go type.
	Parameters jsontext.Value `json:"parameters,omitzero"`
	// Strict asks for arguments that match the schema exactly.
	Strict *bool `json:"strict,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolChoice is the "tool_choice" of a request: a mode ("auto", "none", "required") or a named function. Build
// one with ToolChoiceMode or ToolChoiceFunction.
type ToolChoice struct {
	mode     string
	function *NamedToolChoice
}

// NamedToolChoice is the object form of a ToolChoice.
type NamedToolChoice struct {
	// Type is "function".
	Type string `json:"type"`
	// Function names the function to call.
	Function FunctionName `json:"function"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionName names the function of a NamedToolChoice.
type FunctionName struct {
	// Name is the function name.
	Name string `json:"name"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolChoiceMode returns a tool choice given as a mode: "auto", "none" or "required".
func ToolChoiceMode(mode string) ToolChoice { return ToolChoice{mode: mode} }

// ToolChoiceFunction returns a tool choice that forces a call of the named function.
func ToolChoiceFunction(name string) ToolChoice {
	return ToolChoice{function: &NamedToolChoice{Type: "function", Function: FunctionName{Name: name}}}
}

// Mode returns the mode of a tool choice given as a string.
func (c ToolChoice) Mode() string { return c.mode }

// Function returns the object of a tool choice given as an object.
func (c ToolChoice) Function() *NamedToolChoice { return c.function }

// IsZero reports whether the tool choice is unset.
func (c ToolChoice) IsZero() bool { return c.mode == "" && c.function == nil }

// Equal reports whether two tool choices have the same form and content.
func (c ToolChoice) Equal(o ToolChoice) bool {
	return c.mode == o.mode && reflect.DeepEqual(c.function, o.function)
}

// MarshalJSONTo implements json.MarshalerTo.
func (c ToolChoice) MarshalJSONTo(enc *jsontext.Encoder) error {
	if c.function != nil {
		return json.MarshalEncode(enc, c.function)
	}
	return enc.WriteToken(jsontext.String(c.mode))
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (c *ToolChoice) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '"':
		var s string
		if err := json.UnmarshalDecode(dec, &s); err != nil {
			return err
		}
		*c = ToolChoice{mode: s}
	case '{':
		n := new(NamedToolChoice)
		if err := json.UnmarshalDecode(dec, n); err != nil {
			return err
		}
		*c = ToolChoice{function: n}
	default:
		return fmt.Errorf("openai: tool_choice must be a string or an object, got %s", kind)
	}
	return nil
}

// ResponseFormat is the "response_format" of a request. llama.cpp turns a schema into a grammar; it reads
// JSONSchema.Schema for "json_schema" and Schema for "json_object" or a "json_schema" without the wrapper.
type ResponseFormat struct {
	// Type is "text", "json_object" or "json_schema".
	Type string `json:"type"`
	// JSONSchema is the schema of a "json_schema" format.
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitzero"`
	// Schema is a schema given directly, a llama.cpp extension.
	Schema jsontext.Value `json:"schema,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// JSONSchemaFormat is the schema of a "json_schema" ResponseFormat.
type JSONSchemaFormat struct {
	// Name names the schema.
	Name string `json:"name,omitzero"`
	// Description describes the schema.
	Description *string `json:"description,omitzero"`
	// Schema is the JSON schema; llama.SchemaFor builds one from a Go type.
	Schema jsontext.Value `json:"schema,omitzero"`
	// Strict asks for output that matches the schema exactly.
	Strict *bool `json:"strict,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// StreamOptions are the "stream_options" of a streaming request.
type StreamOptions struct {
	// IncludeUsage adds a final chunk with the token usage.
	IncludeUsage *bool `json:"include_usage,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}
