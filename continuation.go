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
	"fmt"
)

// Continuation selects the final assistant message field to continue. An absent member,
// null and false all mean no continuation unless --prefill-assistant selects automatic
// continuation for a final assistant message.
type Continuation uint8

const (
	// ContinuationUnset omits continue_final_message from a request.
	ContinuationUnset Continuation = iota
	// ContinuationNull sends an explicit null.
	ContinuationNull
	// ContinuationDisabled sends false.
	ContinuationDisabled
	// ContinuationAuto sends true and lets the server select the field.
	ContinuationAuto
	// ContinuationContent continues the content field.
	ContinuationContent
	// ContinuationReasoning continues the reasoning_content field.
	ContinuationReasoning
)

// MarshalJSONTo writes the selected continuation form.
func (c Continuation) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch c {
	case ContinuationNull:
		return enc.WriteToken(jsontext.Null)
	case ContinuationDisabled:
		return enc.WriteToken(jsontext.False)
	case ContinuationAuto:
		return enc.WriteToken(jsontext.True)
	case ContinuationContent:
		return enc.WriteToken(jsontext.String("content"))
	case ContinuationReasoning:
		return enc.WriteToken(jsontext.String("reasoning_content"))
	default:
		return fmt.Errorf("llama: invalid chat continuation %d", c)
	}
}

// UnmarshalJSONFrom reads a boolean, null, or an explicit field name accepted by llama.cpp.
func (c *Continuation) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	t, err := dec.ReadToken()
	if err != nil {
		return err
	}
	switch t.Kind() {
	case 'n':
		*c = ContinuationNull
	case 't':
		*c = ContinuationAuto
	case 'f':
		*c = ContinuationDisabled
	case '"':
		switch t.String() {
		case "content":
			*c = ContinuationContent
		case "reasoning_content":
			*c = ContinuationReasoning
		default:
			return fmt.Errorf("llama: invalid chat continuation field %q", t.String())
		}
	default:
		return fmt.Errorf("llama: expected boolean, string or null chat continuation, got %s", t.Kind())
	}
	return nil
}
