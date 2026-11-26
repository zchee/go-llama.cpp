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
)

// ResponseInput is a string prompt or an ordered list of conversation items. Set exactly one field.
type ResponseInput = responsesTextOrList[ResponseItem]

// ResponseContent is a string or an ordered list of message or tool-output parts. Set exactly one field.
type ResponseContent = responsesTextOrList[ResponsePart]

type responsesTextOrList[T any] struct {
	Text *string
	List []T
}

// MarshalJSONTo writes the selected string or array.
func (v responsesTextOrList[T]) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case v.Text != nil && v.List == nil:
		return json.MarshalEncode(enc, *v.Text)
	case v.Text == nil && v.List != nil:
		return json.MarshalEncode(enc, v.List)
	default:
		return errors.New("expected exactly one of Text or List")
	}
}

// UnmarshalJSONFrom reads a string or an array without changing its form.
func (v *responsesTextOrList[T]) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var next responsesTextOrList[T]
	switch dec.PeekKind() {
	case '"':
		if err := json.UnmarshalDecode(dec, &next.Text); err != nil {
			return err
		}
	case '[':
		if err := json.UnmarshalDecode(dec, &next.List); err != nil {
			return err
		}
	default:
		return errors.New("expected a string or an array")
	}
	*v = next
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
	Type             string                    `json:"type"`
	ID               *string                   `json:"id,omitzero"`
	Status           *string                   `json:"status,omitzero"`
	Summary          []jsontext.Value          `json:"summary,omitzero"`
	Content          []ResponsePart            `json:"content,omitzero"`
	EncryptedContent *string                   `json:"encrypted_content,omitzero"`
	Extra            map[string]jsontext.Value `json:",embed"`
}

// ResponseUnknown preserves the fields of an unrecognized item, part or event without interpreting them.
type ResponseUnknown struct {
	Type  string                    `json:"type"`
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
