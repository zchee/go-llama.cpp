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
	"errors"
	"slices"
)

// ResponseInput is a string prompt or an ordered list of conversation items.
// Construct it with ResponseInputText or ResponseInputItems.
type ResponseInput struct {
	text  *string
	items []ResponseItem
}

// ResponseInputText returns a string prompt, including an empty string.
func ResponseInputText(s string) ResponseInput { return ResponseInput{text: &s} }

// ResponseInputItems returns a conversation encoded as an array, including an empty one.
func ResponseInputItems(items ...ResponseItem) ResponseInput {
	if items == nil {
		items = []ResponseItem{}
	}
	return ResponseInput{items: slices.Clone(items)}
}

// Text returns the prompt and whether the input is a string.
func (i ResponseInput) Text() (string, bool) {
	if i.text == nil {
		return "", false
	}
	return *i.text, true
}

// Items returns the conversation items, or nil if the input is not an array.
func (i ResponseInput) Items() []ResponseItem { return i.items }

// IsZero reports whether the input is unset.
func (i ResponseInput) IsZero() bool { return i.text == nil && i.items == nil }

// MarshalJSONTo writes the selected string or array.
func (i ResponseInput) MarshalJSONTo(enc *jsontext.Encoder) error {
	return writeResponseTextOrList(enc, i.text, i.items)
}

// UnmarshalJSONFrom reads a string or an array of conversation items.
func (i *ResponseInput) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return readResponseTextOrList(dec, &i.text, &i.items)
}

// ResponseContent is a string or an ordered list of message or tool-output parts.
// Construct it with ResponseContentText or ResponseContentParts.
type ResponseContent struct {
	text  *string
	parts []ResponsePart
}

// ResponseContentText returns string content, including an empty string.
func ResponseContentText(s string) ResponseContent { return ResponseContent{text: &s} }

// ResponseContentParts returns content encoded as an array, including an empty one.
func ResponseContentParts(parts ...ResponsePart) ResponseContent {
	if parts == nil {
		parts = []ResponsePart{}
	}
	return ResponseContent{parts: slices.Clone(parts)}
}

// Text returns the text and whether the content is a string.
func (c ResponseContent) Text() (string, bool) {
	if c.text == nil {
		return "", false
	}
	return *c.text, true
}

// Parts returns the content parts, or nil if the content is not an array.
func (c ResponseContent) Parts() []ResponsePart { return c.parts }

// IsZero reports whether the content is unset.
func (c ResponseContent) IsZero() bool { return c.text == nil && c.parts == nil }

// MarshalJSONTo writes the selected string or array.
func (c ResponseContent) MarshalJSONTo(enc *jsontext.Encoder) error {
	return writeResponseTextOrList(enc, c.text, c.parts)
}

// UnmarshalJSONFrom reads a string or an array of content parts.
func (c *ResponseContent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return readResponseTextOrList(dec, &c.text, &c.parts)
}

func writeResponseTextOrList[T any](enc *jsontext.Encoder, text *string, list []T) error {
	if text != nil {
		return enc.WriteToken(jsontext.String(*text))
	}
	if list != nil {
		return json.MarshalEncode(enc, list)
	}
	return errors.New("expected a string or an array")
}

func readResponseTextOrList[T any](dec *jsontext.Decoder, text **string, list *[]T) error {
	switch dec.PeekKind() {
	case '"':
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		*text, *list = new(tok.String()), nil
	case '[':
		var values []T
		if err := json.UnmarshalDecode(dec, &values); err != nil {
			return err
		}
		*text, *list = nil, values
	default:
		return errors.New("expected a string or an array")
	}
	return nil
}

// ResponseItem is one item in the input history or the response output.
// Value is a *ResponseMessage, *ResponseFunctionCall, *ResponseFunctionOutput,
// *ResponseReasoningItem, or *ResponseUnknown.
type ResponseItem struct{ Value ResponseItemValue }

// ResponseItemValue is the object selected by an item's type discriminator.
type ResponseItemValue interface{ responsesItem() }

// ResponseMessage is a message item. Type may be omitted on user, system and developer input.
type ResponseMessage struct {
	Type    string                    `json:"type,omitzero"`
	Role    string                    `json:"role"`
	Content ResponseContent           `json:"content,omitzero"`
	ID      *string                   `json:"id,omitzero"`
	Status  *string                   `json:"status,omitzero"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// ResponseFunctionCall is a function_call item. Arguments is JSON encoded inside a string.
type ResponseFunctionCall struct {
	Type      string                    `json:"type"`
	CallID    string                    `json:"call_id"`
	Name      string                    `json:"name"`
	Arguments string                    `json:"arguments"`
	ID        *string                   `json:"id,omitzero"`
	Status    *string                   `json:"status,omitzero"`
	Extra     map[string]jsontext.Value `json:",embed"`
}

// ResponseFunctionOutput is a function_call_output supplied by the caller in the next request.
type ResponseFunctionOutput struct {
	Type   string                    `json:"type"`
	CallID string                    `json:"call_id"`
	Output ResponseContent           `json:"output,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// ResponseReasoningItem is a reasoning item. The server reads the first content part's text from history.
type ResponseReasoningItem struct {
	Type   string  `json:"type"`
	ID     *string `json:"id,omitzero"`
	Status *string `json:"status,omitzero"`
	// Summary is always an array on the wire so the server can identify reasoning items in history.
	Summary          []jsontext.Value          `json:"summary"`
	Content          []ResponsePart            `json:"content,omitzero"`
	EncryptedContent *string                   `json:"encrypted_content,omitzero"`
	Extra            map[string]jsontext.Value `json:",embed"`
}

// MarshalJSONTo writes the summary as an array, including when it is nil.
func (i ResponseReasoningItem) MarshalJSONTo(enc *jsontext.Encoder) error {
	if i.Summary == nil {
		i.Summary = []jsontext.Value{}
	}
	type wire ResponseReasoningItem
	return json.MarshalEncode(enc, wire(i))
}

// ResponseUnknown preserves the fields of an unrecognized item, part or event without interpreting them.
type ResponseUnknown struct {
	Type  string                    `json:"type,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

func (*ResponseMessage) responsesItem()        {}
func (*ResponseFunctionCall) responsesItem()   {}
func (*ResponseFunctionOutput) responsesItem() {}
func (*ResponseReasoningItem) responsesItem()  {}
func (*ResponseUnknown) responsesItem()        {}

// MarshalJSONTo writes the selected item's object.
func (i ResponseItem) MarshalJSONTo(enc *jsontext.Encoder) error {
	return json.MarshalEncode(enc, i.Value)
}

// UnmarshalJSONFrom selects the item by type; untagged input messages use the message shape.
func (i *ResponseItem) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return responsesReadTagged(dec, &i.Value, func(kind string) ResponseItemValue {
		switch kind {
		case "message", "":
			return new(ResponseMessage)
		case "function_call":
			return new(ResponseFunctionCall)
		case "function_call_output":
			return new(ResponseFunctionOutput)
		case "reasoning":
			return new(ResponseReasoningItem)
		default:
			return new(ResponseUnknown)
		}
	})
}

// ResponsePart is one content part. Value is a *ResponseTextPart, *ResponseImagePart,
// *ResponseRefusalPart, or *ResponseUnknown.
type ResponsePart struct{ Value ResponsePartValue }

// ResponsePartValue is the object selected by a content part's type discriminator.
type ResponsePartValue interface{ responsesPart() }

// ResponseTextPart is an input_text, output_text or reasoning_text part. The server emits empty
// annotations and logprobs arrays on finished output_text parts but not on unfinished ones.
type ResponseTextPart struct {
	Type        string                    `json:"type"`
	Text        string                    `json:"text"`
	Annotations []jsontext.Value          `json:"annotations,omitzero"`
	Logprobs    []jsontext.Value          `json:"logprobs,omitzero"`
	Extra       map[string]jsontext.Value `json:",embed"`
}

// ResponseImagePart is an input_image part. ImageURL is a string, not a chat image_url object.
type ResponseImagePart struct {
	Type     string                    `json:"type"`
	ImageURL string                    `json:"image_url"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// ResponseRefusalPart is a refusal in an assistant message supplied as history.
type ResponseRefusalPart struct {
	Type    string                    `json:"type"`
	Refusal string                    `json:"refusal"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

func (*ResponseTextPart) responsesPart()    {}
func (*ResponseImagePart) responsesPart()   {}
func (*ResponseRefusalPart) responsesPart() {}
func (*ResponseUnknown) responsesPart()     {}

// MarshalJSONTo writes the selected part's object.
func (p ResponsePart) MarshalJSONTo(enc *jsontext.Encoder) error {
	return json.MarshalEncode(enc, p.Value)
}

// UnmarshalJSONFrom selects the part by type and retains unknown kinds.
func (p *ResponsePart) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return responsesReadTagged(dec, &p.Value, func(kind string) ResponsePartValue {
		switch kind {
		case "input_text", "output_text", "reasoning_text":
			return new(ResponseTextPart)
		case "input_image":
			return new(ResponseImagePart)
		case "refusal":
			return new(ResponseRefusalPart)
		default:
			return new(ResponseUnknown)
		}
	})
}

// The discriminator can follow the payload. Read the object first rather than assuming member order.
func responsesReadTagged[T any](dec *jsontext.Decoder, dst *T, selectKind func(string) T) error {
	if dec.PeekKind() != '{' {
		return errors.New("expected an object with a type discriminator")
	}
	raw, err := dec.ReadValue()
	if err != nil {
		return err
	}
	var tag struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &tag); err != nil {
		return err
	}
	v := selectKind(tag.Type)
	if err := json.Unmarshal(raw, v, dec.Options()); err != nil {
		return err
	}
	*dst = v
	return nil
}
