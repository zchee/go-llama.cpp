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
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
)

// TokenizeRequest is the body sent to POST /tokenize.
type TokenizeRequest struct {
	// Content is a string or a single sequence of strings and token IDs; omission returns an empty token list.
	Content TokenizeContent `json:"content,omitzero"`
	// AddSpecial inserts model-specific special tokens; the server defaults to false.
	// For array content, the first element must be text for this to take effect.
	AddSpecial *bool `json:"add_special,omitzero"`
	// ParseSpecial recognizes special tokens in text; the server defaults to true.
	ParseSpecial *bool `json:"parse_special,omitzero"`
	// WithPieces returns each token's spelling as well as its ID; the server defaults to false.
	WithPieces *bool `json:"with_pieces,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// TokenizeContent is a string or an array of strings and token IDs. Unlike Prompt, it never represents a batch.
// Construct it with TokenizeText or TokenizeParts. The zero value is omitted from TokenizeRequest.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type TokenizeContent struct {
	text  *string
	parts []PromptPart
}

// TokenizeText returns content encoded as a single string.
func TokenizeText(text string) TokenizeContent { return TokenizeContent{text: &text} }

// TokenizeParts returns content encoded as an array. Use TextPart and TokenPart to build its elements.
func TokenizeParts(parts ...PromptPart) TokenizeContent {
	return TokenizeContent{parts: append([]PromptPart{}, parts...)}
}

// Text returns the text and whether the content is a single string.
func (c TokenizeContent) Text() (string, bool) {
	if c.text == nil {
		return "", false
	}
	return *c.text, true
}

// Parts returns a read-only view of the array elements, or nil if the content is not an array.
func (c TokenizeContent) Parts() []PromptPart { return c.parts }

// IsZero reports whether the content is unset.
func (c TokenizeContent) IsZero() bool { return c.text == nil && c.parts == nil }

// MarshalJSONTo implements json.MarshalerTo.
func (c TokenizeContent) MarshalJSONTo(enc *jsontext.Encoder) error {
	if c.text != nil {
		return enc.WriteToken(jsontext.String(*c.text))
	}
	if c.parts == nil {
		return errors.New("llama: tokenize content is unset")
	}
	return writePromptParts(enc, c.parts)
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (c *TokenizeContent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '"':
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		*c = TokenizeText(tok.String())
	case '[':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		parts := []PromptPart{}
		for kind := dec.PeekKind(); kind != ']'; kind = dec.PeekKind() {
			switch kind {
			case '"':
				tok, err := dec.ReadToken()
				if err != nil {
					return err
				}
				parts = append(parts, TextPart(tok.String()))
			case '0':
				var id Token
				if err := json.UnmarshalDecode(dec, &id); err != nil {
					return err
				}
				parts = append(parts, TokenPart(id))
			default:
				return fmt.Errorf("llama: tokenize content element must be a string or token ID, got %s", kind)
			}
		}
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*c = TokenizeContent{parts: parts}
	default:
		return fmt.Errorf("llama: tokenize content must be a string or array, got %s", kind)
	}
	return nil
}

// TokenizeResponse is the JSON returned by POST /tokenize.
type TokenizeResponse struct {
	// Tokens is the model's token sequence for the request content.
	Tokens []TokenizeToken `json:"tokens"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// TokenizeToken is a bare token ID when Piece is nil, or an object with an ID and piece otherwise.
type TokenizeToken struct {
	// ID is the token's position in the model's vocabulary.
	ID Token `json:"id"`
	// Piece is the token's spelling as text or raw bytes; nil unless the request set WithPieces.
	Piece *TokenPiece `json:"piece"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// TokenizeID returns a bare token ID.
func TokenizeID(id Token) TokenizeToken { return TokenizeToken{ID: id} }

// TokenizeIDWithPiece returns a token ID with its text or byte representation.
func TokenizeIDWithPiece(id Token, piece TokenPiece) TokenizeToken {
	return TokenizeToken{ID: id, Piece: &piece}
}

type tokenizeTokenObject TokenizeToken

// MarshalJSONTo implements json.MarshalerTo.
func (t TokenizeToken) MarshalJSONTo(enc *jsontext.Encoder) error {
	if t.Piece != nil {
		return json.MarshalEncode(enc, tokenizeTokenObject(t))
	}
	if len(t.Extra) != 0 {
		return errors.New("llama: a bare tokenize token ID cannot have extra fields")
	}
	return enc.WriteToken(jsontext.Int(int64(t.ID)))
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (t *TokenizeToken) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '0':
		var id Token
		if err := json.UnmarshalDecode(dec, &id); err != nil {
			return err
		}
		*t = TokenizeID(id)
	case '{':
		var obj tokenizeTokenObject
		if err := json.UnmarshalDecode(dec, &obj); err != nil {
			return err
		}
		if obj.Piece == nil {
			return errors.New("llama: tokenize token object must have a piece")
		}
		*t = TokenizeToken(obj)
	default:
		return fmt.Errorf("llama: tokenize token must be an ID or object, got %s", kind)
	}
	return nil
}

// TokenPiece is a token's UTF-8 string or, for an invalid UTF-8 fragment, a JSON array of byte values.
// Construct it with TokenPieceText or TokenPieceBytes. The zero value represents an empty string.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type TokenPiece struct {
	text string
	raw  []byte
}

// TokenPieceText returns a piece encoded as a JSON string.
func TokenPieceText(text string) TokenPiece { return TokenPiece{text: text} }

// TokenPieceBytes returns a piece encoded as an array of byte values, not a base64 string.
func TokenPieceBytes(data []byte) TokenPiece {
	return TokenPiece{raw: append([]byte{}, data...)}
}

// Text returns the text and whether the piece was encoded as a string.
func (p TokenPiece) Text() (string, bool) { return p.text, p.raw == nil }

// Bytes returns a read-only view of the bytes of an array piece, or nil for a text piece.
func (p TokenPiece) Bytes() []byte { return p.raw }

// MarshalJSONTo implements json.MarshalerTo.
func (p TokenPiece) MarshalJSONTo(enc *jsontext.Encoder) error {
	if p.raw == nil {
		return enc.WriteToken(jsontext.String(p.text))
	}
	if err := enc.WriteToken(jsontext.BeginArray); err != nil {
		return err
	}
	for _, b := range p.raw {
		if err := enc.WriteToken(jsontext.Uint(uint64(b))); err != nil {
			return err
		}
	}
	return enc.WriteToken(jsontext.EndArray)
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (p *TokenPiece) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '"':
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		*p = TokenPieceText(tok.String())
	case '[':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		raw := []byte{}
		for kind := dec.PeekKind(); kind != ']'; kind = dec.PeekKind() {
			if kind != '0' {
				return fmt.Errorf("llama: tokenize piece byte must be a number, got %s", kind)
			}
			var b byte
			if err := json.UnmarshalDecode(dec, &b); err != nil {
				return err
			}
			raw = append(raw, b)
		}
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*p = TokenPiece{raw: raw}
	default:
		return fmt.Errorf("llama: tokenize piece must be a string or an array of bytes, got %s", kind)
	}
	return nil
}

// Tokenize converts text and existing token IDs to the model's token sequence.
func (c *Client) Tokenize(ctx context.Context, req *TokenizeRequest, opts ...RequestOption) (*TokenizeResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil tokenize request")
	}
	return Call[*TokenizeResponse](ctx, c, http.MethodPost, "/tokenize", req, opts...)
}
