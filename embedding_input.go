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
	"errors"
	"fmt"
	"slices"
)

// EmbeddingInput is text, token ids, mixed text and ids, a prompt or content object, or a batch of inputs.
// Its zero value is unset.
type EmbeddingInput struct {
	prompt  Prompt
	object  *EmbeddingPromptObject
	content *EmbeddingContent
	batch   []EmbeddingInput
}

// EmbeddingPromptObject carries a native prompt with optional base64-encoded media files.
type EmbeddingPromptObject struct {
	// PromptString is text, or without media, a list of text and/or token ids.
	// Use BatchPrompt of TextPrompt values for a list of only text.
	PromptString   Prompt   `json:"prompt_string"`
	MultimodalData [][]byte `json:"multimodal_data,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// EmbeddingContent carries chat-style content parts for a single input.
type EmbeddingContent struct {
	Content []ContentPart `json:"content"`

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

// Batch returns the batch of inputs, or nil for other shapes.
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
		if !slices.ContainsFunc(elems, isNumber) {
			batch := make([]EmbeddingInput, len(elems))
			for i, elem := range elems {
				if err := batch[i].unmarshalEmbeddingEntry(elem); err != nil {
					return fmt.Errorf("llama: embedding input %d: %w", i, err)
				}
				if batch[i].IsZero() {
					return fmt.Errorf("llama: embedding input %d is null", i)
				}
			}
			*in = EmbeddingInput{batch: batch}
			return nil
		}
	}
	return in.unmarshalEmbeddingEntry(v)
}

func (in *EmbeddingInput) unmarshalEmbeddingEntry(v jsontext.Value) error {
	var result EmbeddingInput
	if v.Kind() == '{' {
		var fields map[string]jsontext.Value
		if err := json.Unmarshal(v, &fields); err != nil {
			return err
		}
		if content, ok := fields["content"]; ok {
			if content.Kind() != '[' {
				return errors.New("llama: embedding content must be an array")
			}
			if err := json.Unmarshal(v, &result.content, jsonOptions); err != nil {
				return err
			}
		} else if _, ok := fields["prompt_string"]; ok {
			if err := json.Unmarshal(v, &result.object, jsonOptions); err != nil {
				return err
			}
		} else {
			return errors.New("llama: embedding input object needs content or prompt_string")
		}
	} else if err := result.prompt.unmarshalSingle(v); err != nil {
		return err
	}
	*in = result
	return nil
}
