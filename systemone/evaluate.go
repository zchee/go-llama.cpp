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

package systemone

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
)

// PayloadKind is the JSON shape of a Payload.
type PayloadKind int

const (
	// PayloadKindNone is an unset payload.
	PayloadKindNone PayloadKind = iota
	// PayloadKindText is a JSON string.
	PayloadKindText
	// PayloadKindNull is an explicit JSON null.
	PayloadKindNull
	// PayloadKindObject is a JSON object.
	PayloadKindObject
	// PayloadKindArray is a JSON array.
	PayloadKindArray
	// PayloadKindRaw is a JSON number or boolean.
	PayloadKindRaw
)

// Payload is a state, instruction or description. Its zero value is omitted from optional fields;
// Null sends an explicit JSON null. Support for null, numbers and booleans depends on the model.
type Payload struct {
	text string
	raw  jsontext.Value
	kind jsontext.Kind
}

// Text returns a string payload, including an empty string.
func Text(s string) Payload { return Payload{text: s, kind: '"'} }

// Null returns an explicit JSON null.
func Null() Payload { return Payload{kind: 'n'} }

// Raw returns a payload from an encoded JSON value, preserving its JSON representation.
func Raw(v jsontext.Value) (Payload, error) {
	var p Payload
	if err := json.Unmarshal(v, &p); err != nil {
		return Payload{}, err
	}
	return p, nil
}

// NewPayload encodes a Go value as a payload. For example, a slice of openai.Message
// encodes a conversation; an openai.ContentPart can encode a spoken instruction.
func NewPayload(v any) (Payload, error) {
	data, err := json.Marshal(v, json.Deterministic(true))
	if err != nil {
		return Payload{}, err
	}
	var p Payload
	err = json.Unmarshal(data, &p)
	return p, err
}

// Kind returns the shape of the payload.
func (p Payload) Kind() PayloadKind {
	switch p.kind {
	case 0:
		return PayloadKindNone
	case '"':
		return PayloadKindText
	case 'n':
		return PayloadKindNull
	case '{':
		return PayloadKindObject
	case '[':
		return PayloadKindArray
	default:
		return PayloadKindRaw
	}
}

// Text returns the string and whether the payload is text.
func (p Payload) Text() (string, bool) { return p.text, p.kind == '"' }

// IsNull reports whether the payload is explicitly null.
func (p Payload) IsNull() bool { return p.kind == 'n' }

// Raw returns a copy of the encoded object, array, number or boolean, or nil for any other kind.
func (p Payload) Raw() jsontext.Value { return p.raw.Clone() }

// IsZero reports whether the payload is unset, as distinct from an explicit null.
func (p Payload) IsZero() bool { return p.kind == 0 }

// MarshalJSONTo implements json.MarshalerTo.
func (p Payload) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch p.kind {
	case '"':
		return enc.WriteToken(jsontext.String(p.text))
	case 'n':
		return enc.WriteToken(jsontext.Null)
	case '{', '[', '0', 't', 'f':
		return enc.WriteValue(p.raw)
	default:
		return errors.New("systemone: payload is unset; use Null for a JSON null")
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (p *Payload) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	kind := dec.PeekKind()
	switch kind {
	case '"':
		token, err := dec.ReadToken()
		if err != nil {
			return err
		}
		*p = Text(token.String())
	case 'n':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*p = Null()
	default:
		raw, err := dec.ReadValue()
		if err != nil {
			return err
		}
		*p = Payload{kind: kind, raw: raw.Clone()}
	}
	return nil
}

// Request asks named questions about State. Model selects the target model in router mode;
// single-model servers ignore it. Files and Images are alternative spellings for a list of media
// data URLs; set at most one. Extra cannot override any named request field.
type Request struct {
	// Model selects the target in router mode and is ignored by a single-model server.
	Model string `json:"model,omitzero"`
	// State is required text or JSON to evaluate; some models accept explicit null for media-only input.
	State Payload `json:"state"`
	// Questions is a required, nonempty map from caller-chosen IDs to questions about the state.
	Questions map[string]Question `json:"questions"`
	// Files supplies optional media data URLs to place before State, subject to the model's media and count limits.
	Files []string `json:"files,omitzero"`
	// Images is an optional alias for Files, which the client does not allow in the same request.
	Images []string `json:"images,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// MarshalJSONTo implements json.MarshalerTo.
func (r Request) MarshalJSONTo(enc *jsontext.Encoder) error {
	if r.Files != nil && r.Images != nil {
		return errors.New("systemone: set only one of files and images")
	}
	if err := reserved(r.Extra, "model", "state", "questions", "files", "images"); err != nil {
		return err
	}
	type wire Request
	return json.MarshalEncode(enc, wire(r))
}

func reserved(extra map[string]jsontext.Value, names ...string) error {
	for _, name := range names {
		if _, ok := extra[name]; ok {
			return fmt.Errorf("systemone: %q is reserved and cannot be supplied in Extra", name)
		}
	}
	return nil
}

// Response contains the loaded model's answers, keyed by question ID.
type Response struct {
	// Model is the server-reported name of the decision model that answered the request.
	Model string `json:"model"`
	// Answers maps the request's question IDs to results of their respective types.
	Answers map[string]Answer `json:"answers"`
	// Usage reports token counts for the whole request, not individual answers.
	Usage Usage `json:"usage"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Usage counts prompt tokens across the questions. OutputTokens is zero for decision models.
type Usage struct {
	// InputTokens counts prompt tokens for all answered questions.
	InputTokens int `json:"input_tokens"`
	// OutputTokens is always 0 because the decision endpoint does not generate tokens.
	OutputTokens int `json:"output_tokens"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Evaluate answers the questions in req using POST /v1/systemone. This endpoint does not stream.
func (c *Client) Evaluate(ctx context.Context, req *Request, opts ...llama.RequestOption) (*Response, error) {
	if req == nil {
		return nil, errors.New("systemone: nil evaluate request")
	}
	return llama.Call[*Response](ctx, c.c, http.MethodPost, "/v1/systemone", req, opts...)
}
