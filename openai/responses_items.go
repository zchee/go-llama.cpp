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

// Items returns a read-only view of the stored conversation items, or nil if the input is not an array.
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

// Parts returns a read-only view of the stored content parts, or nil if the content is not an array.
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
type ResponseItem struct {
	// Value holds the concrete object selected by the wire type, or by the message shape for untagged input.
	Value ResponseItemValue
}

// ResponseItemValue is the object selected by an item's type discriminator.
type ResponseItemValue interface{ responsesItem() }

// ResponseMessage is a message item. Type may be omitted on user, system and developer input.
type ResponseMessage struct {
	// Type is "message" in output and assistant history; user, system, and developer messages may omit it.
	Type string `json:"type,omitzero"`
	// Role is "assistant" in output and identifies the speaker for each input message.
	Role string `json:"role"`
	// Content carries text or parts; assistant history without content is converted to an empty part list.
	Content ResponseContent `json:"content,omitzero"`
	// ID identifies the server-generated assistant message across its stream events and is not needed in history.
	ID *string `json:"id,omitzero"`
	// Status is "in_progress" on an added output message and "completed" on a finished one; history ignores it.
	Status *string `json:"status,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseFunctionCall is a function_call item. Arguments is JSON encoded inside a string.
type ResponseFunctionCall struct {
	// Type is "function_call" in output and is required when supplying the call in history.
	Type string `json:"type"`
	// CallID is the call identifier to repeat in the matching ResponseFunctionOutput.
	CallID string `json:"call_id"`
	// Name identifies the declared function the model chose to call.
	Name string `json:"name"`
	// Arguments is a string containing the JSON arguments; it is empty when the stream first adds the call.
	Arguments string `json:"arguments"`
	// ID is the server-assigned item identifier used in stream events, distinct from CallID and ignored in history.
	ID *string `json:"id,omitzero"`
	// Status is "in_progress" on an added call and "completed" when finished; the server ignores it in history.
	Status *string `json:"status,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseFunctionOutput is a function_call_output supplied by the caller in the next request.
type ResponseFunctionOutput struct {
	// Type must be "function_call_output" so the server converts this item to a tool message.
	Type string `json:"type"`
	// CallID must match the CallID of the function call this result answers.
	CallID string `json:"call_id"`
	// Output supplies the required tool result as a string or an array of input_text and input_image parts.
	Output ResponseContent `json:"output,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseReasoningItem is a reasoning item. The server reads the first content part's text from history.
type ResponseReasoningItem struct {
	// Type is "reasoning" and is required on a reasoning item supplied in history.
	Type string `json:"type"`
	// ID is the server-assigned reasoning item identifier used in stream deltas and ignored in history.
	ID *string `json:"id,omitzero"`
	// Status is "in_progress" when a stream adds the item, "completed" without streaming, and absent from the
	// final stream item.
	Status *string `json:"status,omitzero"`
	// Summary is always an array on the wire so the server can identify reasoning items in history.
	Summary []jsontext.Value `json:"summary"`
	// Content holds the reasoning text, is empty when a stream adds the item, and must be nonempty in history.
	Content []ResponsePart `json:"content,omitzero"`
	// EncryptedContent is emitted as an empty string and is not read when the item is supplied in history.
	EncryptedContent *string `json:"encrypted_content,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
	// Type retains the unrecognized discriminator so the object can be re-encoded without interpreting it.
	Type string `json:"type,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
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
type ResponsePart struct {
	// Value holds the concrete content object selected by the wire type.
	Value ResponsePartValue
}

// ResponsePartValue is the object selected by a content part's type discriminator.
type ResponsePartValue interface{ responsesPart() }

// ResponseTextPart is an input_text, output_text or reasoning_text part. The server emits empty
// annotations and logprobs arrays on finished output_text parts but not on unfinished ones.
type ResponseTextPart struct {
	// Type distinguishes "input_text", "output_text", and "reasoning_text" on the wire.
	Type string `json:"type"`
	// Text carries the literal prompt, generated reply, or reasoning text according to Type.
	Text string `json:"text"`
	// Annotations is an empty array on completed output text and is otherwise omitted by the server.
	Annotations []jsontext.Value `json:"annotations,omitzero"`
	// Logprobs is an empty array on completed output text; the server does not populate token probabilities here.
	Logprobs []jsontext.Value `json:"logprobs,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseImagePart is an input_image part. ImageURL is a string, not a chat image_url object.
type ResponseImagePart struct {
	// Type must be "input_image" to pass the image to the chat parser.
	Type string `json:"type"`
	// ImageURL is the required image source, supplied as an HTTP(S) URL, data URL, or bare base64.
	ImageURL string `json:"image_url"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseRefusalPart is a refusal in an assistant message supplied as history.
type ResponseRefusalPart struct {
	// Type must be "refusal" to identify this part in assistant history.
	Type string `json:"type"`
	// Refusal supplies the required text of the prior assistant refusal.
	Refusal string `json:"refusal"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
