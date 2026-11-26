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
	// InputTokens is the number of tokens the input would take.
	InputTokens int64 `json:"input_tokens"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// Tool is a tool the model may call; see llama.Tool.
type Tool = llama.Tool

// FunctionDefinition describes a function tool; see llama.FunctionDefinition.
type FunctionDefinition = llama.FunctionDefinition

// ToolChoice is a mode, a named function, or an allowlist of functions; see llama.ToolChoice.
type ToolChoice = llama.ToolChoice

// NamedToolChoice is the object form of a ToolChoice that names one function; see llama.NamedToolChoice.
type NamedToolChoice = llama.NamedToolChoice

// FunctionName names the function of a NamedToolChoice; see llama.FunctionName.
type FunctionName = llama.FunctionName

// AllowedToolChoice is the allowlist form of a ToolChoice; see llama.AllowedToolChoice.
type AllowedToolChoice = llama.AllowedToolChoice

// AllowedTools holds the selection mode and function references of an AllowedToolChoice; see llama.AllowedTools.
type AllowedTools = llama.AllowedTools

// ToolChoiceMode returns a tool choice given as a mode: "auto", "none" or "required".
func ToolChoiceMode(mode string) ToolChoice { return llama.ToolChoiceMode(mode) }

// ToolChoiceFunction returns a tool choice that forces a call of the named function.
func ToolChoiceFunction(name string) ToolChoice { return llama.ToolChoiceFunction(name) }

// ToolChoiceAllowedTools returns a tool choice that limits the model to the named functions in the given mode.
func ToolChoiceAllowedTools(mode string, names ...string) ToolChoice {
	return llama.ToolChoiceAllowedTools(mode, names...)
}

// ResponseFormat is the "response_format" of a request; see llama.ResponseFormat.
type ResponseFormat = llama.ResponseFormat

// JSONSchemaFormat is the schema of a "json_schema" ResponseFormat; see llama.JSONSchemaFormat.
type JSONSchemaFormat = llama.JSONSchemaFormat

// StreamOptions are the "stream_options" of a streaming request.
type StreamOptions struct {
	// IncludeUsage adds a final chunk with the token usage.
	IncludeUsage *bool `json:"include_usage,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}
