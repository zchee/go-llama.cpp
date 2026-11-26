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
	"bytes"
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strings"
)

// EmbeddingInput is text, token ids, mixed text and ids, a prompt or content object, or a batch of inputs.
// Its zero value is unset.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type EmbeddingInput struct {
	prompt  Prompt
	object  *EmbeddingPromptObject
	content *EmbeddingContent
	batch   []EmbeddingInput
}

// EmbeddingPromptObject carries a native prompt with optional base64-encoded media files.
type EmbeddingPromptObject struct {
	// PromptString is text, or without media, a list of text and/or token ids.
	// Build it with TokenizeText or TokenizeParts.
	PromptString TokenizeContent `json:"prompt_string"`
	// MultimodalData supplies media files for the prompt's media markers in order, encoded as base64 when present.
	MultimodalData [][]byte `json:"multimodal_data,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

func (o EmbeddingPromptObject) validate() error {
	if _, text := o.PromptString.Text(); !text && o.MultimodalData != nil {
		return errors.New("llama: prompt_string must be a string when multimodal_data is present")
	}
	return nil
}

// MarshalJSONTo encodes the prompt object, rejecting nested prompts and non-string media prompts.
func (o EmbeddingPromptObject) MarshalJSONTo(enc *jsontext.Encoder) error {
	if err := o.validate(); err != nil {
		return err
	}
	type plain EmbeddingPromptObject
	return json.MarshalEncode(enc, plain(o))
}

// UnmarshalJSONFrom decodes one native prompt object without discarding explicit null media.
func (o *EmbeddingPromptObject) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var fields map[string]jsontext.Value
	if err := json.UnmarshalDecode(dec, &fields); err != nil {
		return err
	}
	var out EmbeddingPromptObject
	if err := json.Unmarshal(fields["prompt_string"], &out.PromptString, embeddingTokenDecoder); err != nil {
		return err
	}
	if media, ok := fields["multimodal_data"]; ok {
		if media.Kind() != '[' {
			return errors.New("llama: multimodal_data must be an array of base64 strings")
		}
		var files []*string
		if err := json.Unmarshal(media, &files); err != nil {
			return err
		}
		out.MultimodalData = make([][]byte, len(files))
		for i, file := range files {
			if file == nil || strings.ContainsAny(*file, "\r\n") {
				return fmt.Errorf("llama: multimodal_data[%d] must be a base64 string", i)
			}
			data, err := base64.StdEncoding.Strict().DecodeString(*file)
			if err != nil {
				return fmt.Errorf("llama: multimodal_data[%d]: %w", i, err)
			}
			out.MultimodalData[i] = data
		}
	}
	if err := out.validate(); err != nil {
		return err
	}
	delete(fields, "prompt_string")
	delete(fields, "multimodal_data")
	out.Extra = fields
	*o = out
	return nil
}

// EmbeddingContent carries chat-style content parts for a single input.
type EmbeddingContent struct {
	// Content supplies the ordered chat-style text and media parts to embed as one input.
	Content []ContentPart `json:"content"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingText returns one text input.
func EmbeddingText(text string) EmbeddingInput {
	return EmbeddingInput{prompt: TextPrompt(text)}
}

// EmbeddingTexts returns a batch of text inputs.
func EmbeddingTexts(texts ...string) EmbeddingInput {
	batch := make([]EmbeddingInput, len(texts))
	for i, text := range texts {
		batch[i] = EmbeddingText(text)
	}
	return EmbeddingInput{batch: batch}
}

// EmbeddingTokens returns one input consisting of token ids. Supply at least one id.
func EmbeddingTokens(tokens ...Token) EmbeddingInput {
	return EmbeddingInput{prompt: TokenPrompt(tokens...)}
}

// EmbeddingMixed returns one input mixing strings and token ids. Supply at least one of each.
func EmbeddingMixed(parts ...PromptPart) EmbeddingInput {
	return EmbeddingInput{prompt: MixedPrompt(parts...)}
}

// EmbeddingObject returns one native prompt object.
func EmbeddingObject(object EmbeddingPromptObject) EmbeddingInput {
	return EmbeddingInput{object: &object}
}

// EmbeddingMultimodal returns one input built from chat-style content parts.
func EmbeddingMultimodal(parts ...ContentPart) EmbeddingInput {
	return EmbeddingInput{content: &EmbeddingContent{Content: slices.Clone(parts)}}
}

// EmbeddingBatch returns a batch of inputs. The elements must not be batches or unset.
func EmbeddingBatch(inputs ...EmbeddingInput) EmbeddingInput {
	batch := make([]EmbeddingInput, len(inputs))
	copy(batch, inputs)
	return EmbeddingInput{batch: batch}
}

// Prompt returns the text, token ids, or mixed prompt, or an unset Prompt for other shapes.
func (in EmbeddingInput) Prompt() Prompt { return in.prompt }

// Object returns the native prompt object, or nil for other shapes.
func (in EmbeddingInput) Object() *EmbeddingPromptObject { return in.object }

// Content returns the chat-style content object, or nil for other shapes.
func (in EmbeddingInput) Content() *EmbeddingContent { return in.content }

// Batch returns a read-only view of the batch of inputs, or nil for other shapes.
func (in EmbeddingInput) Batch() []EmbeddingInput { return in.batch }

// IsZero reports whether the input is unset.
func (in EmbeddingInput) IsZero() bool {
	return in.prompt.IsZero() && in.object == nil && in.content == nil && in.batch == nil
}

// MarshalJSONTo implements json.MarshalerTo.
func (in EmbeddingInput) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case in.batch != nil:
		for i := range in.batch {
			sub := &in.batch[i]
			if sub.batch != nil || sub.IsZero() {
				return errors.New("llama: an embedding batch cannot contain a batch or an unset input")
			}
		}
		return json.MarshalEncode(enc, in.batch)
	case in.content != nil:
		return json.MarshalEncode(enc, in.content)
	case in.object != nil:
		return json.MarshalEncode(enc, in.object)
	default:
		return in.prompt.MarshalJSONTo(enc)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom. An array with a top-level token id is one input;
// an array without one is a batch.
func (in *EmbeddingInput) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	v, err := dec.ReadValue()
	if err != nil {
		return err
	}
	if v.Kind() == '[' {
		var elems []jsontext.Value
		if err := json.Unmarshal(v, &elems); err != nil {
			return err
		}
		if slices.ContainsFunc(elems, isNumber) {
			p, err := embeddingArrayPrompt(v, elems)
			if err != nil {
				return err
			}
			*in = EmbeddingInput{prompt: p}
			return nil
		}
		batch := make([]EmbeddingInput, len(elems))
		for i, elem := range elems {
			if err := batch[i].unmarshalEmbeddingEntry(elem); err != nil {
				return fmt.Errorf("llama: embedding input %d: %w", i, err)
			}
		}
		*in = EmbeddingInput{batch: batch}
		return nil
	}
	return in.unmarshalEmbeddingEntry(v)
}

func (in *EmbeddingInput) unmarshalEmbeddingEntry(v jsontext.Value) error {
	var result EmbeddingInput
	switch v.Kind() {
	case '{':
		var fields map[string]jsontext.Value
		if err := json.Unmarshal(v, &fields); err != nil {
			return err
		}
		if content, ok := fields["content"]; ok {
			if content.Kind() != '[' {
				return errors.New("llama: embedding content must be an array")
			}
			var parts []ContentPart
			if err := json.Unmarshal(content, &parts, jsonOptions); err != nil {
				return err
			}
			// Optional fields on shared chat types cannot distinguish some null and
			// empty values from absence. Do not re-encode a different request.
			if bytes.Contains(content, []byte("null")) || bytes.Contains(content, []byte(`""`)) {
				raw, err := json.Marshal(parts, jsonOptions)
				if err != nil {
					return err
				}
				encoded := jsontext.Value(raw)
				for _, v := range []*jsontext.Value{&content, &encoded} {
					if err := v.Format(jsontext.ReorderRawObjects(true)); err != nil {
						return err
					}
				}
				if !bytes.Equal(content, encoded) {
					return errors.New("llama: embedding content contains a null or empty field that cannot be preserved")
				}
			}
			delete(fields, "content")
			result.content = &EmbeddingContent{Content: parts, Extra: fields}
		} else if _, ok := fields["prompt_string"]; ok {
			if err := json.Unmarshal(v, &result.object, jsonOptions); err != nil {
				return err
			}
		} else {
			return errors.New("llama: embedding input object needs content or prompt_string")
		}
	case '[':
		var elems []jsontext.Value
		if err := json.Unmarshal(v, &elems); err != nil {
			return err
		}
		p, err := embeddingArrayPrompt(v, elems)
		if err != nil {
			return err
		}
		result.prompt = p
	case '"':
		if err := result.prompt.unmarshalSingle(v); err != nil {
			return err
		}
	default:
		return errors.New("llama: embedding input must be a string, array or object")
	}
	*in = result
	return nil
}

// Outside prompt_string the server selects a single token sequence by integer spelling,
// not by numeric value. Decimal spellings are accepted only inside a mixed sequence.
func embeddingArrayPrompt(v jsontext.Value, elems []jsontext.Value) (Prompt, error) {
	var hasInteger, hasText bool
	for _, e := range elems {
		hasInteger = hasInteger || (isNumber(e) && !bytes.ContainsAny(e, ".eE"))
		hasText = hasText || e.Kind() == '"'
	}
	if !hasInteger {
		return Prompt{}, errors.New("llama: embedding array must contain at least one integer token")
	}
	if !hasText {
		var tokens []Token
		for _, e := range elems {
			if !isNumber(e) {
				return Prompt{}, errors.New("llama: embedding token list must contain only integer tokens")
			}
		}
		if err := json.Unmarshal(v, &tokens); err != nil {
			return Prompt{}, err
		}
		return Prompt{kind: PromptKindTokens, tokens: tokens}, nil
	}
	var flat TokenizeContent
	if err := json.Unmarshal(v, &flat, embeddingTokenDecoder); err != nil {
		return Prompt{}, err
	}
	return Prompt{kind: PromptKindMixed, parts: flat.Parts()}, nil
}

var embeddingTokenDecoder = json.WithUnmarshalers(json.UnmarshalFunc(func(raw []byte, id *Token) error {
	if !bytes.ContainsAny(raw, ".eE") {
		return json.Unmarshal(raw, id)
	}
	n, ok := new(big.Rat).SetString(string(raw))
	if !ok || !n.IsInt() || !n.Num().IsInt64() || n.Num().Int64() < math.MinInt32 || n.Num().Int64() > math.MaxInt32 {
		return errors.New("llama: embedding token must be an integral int32")
	}
	*id = Token(n.Num().Int64())
	return nil
}))
