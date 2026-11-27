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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
)

// ChoiceCriteria maps option names to descriptions. Use Null for an undescribed option.
type ChoiceCriteria map[string]Payload

// ScoreCriteria lists level descriptions in increasing order.
type ScoreCriteria []Payload

// NoulCriteria describes the false and true alternatives. Some models also accept No and Yes.
type NoulCriteria struct {
	// False describes the negative answer; if omitted, the server can use No on models that accept it.
	False Payload `json:"false,omitzero"`
	// True describes the positive answer; if omitted, the server can use Yes on models that accept it.
	True Payload `json:"true,omitzero"`
	// No is an optional alias for False on lfm2-d1-omni and h2o-lightning, ignored when False is present.
	No Payload `json:"no,omitzero"`
	// Yes is an optional alias for True on lfm2-d1-omni and h2o-lightning, ignored when True is present.
	Yes Payload `json:"yes,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Criteria is the set of possible answers for a question: ChoiceCriteria, ScoreCriteria or *NoulCriteria.
type Criteria interface{ isCriteria() }

func (ChoiceCriteria) isCriteria() {}
func (ScoreCriteria) isCriteria()  {}
func (*NoulCriteria) isCriteria()  {}

// Question asks for one choice, an ordinal score, or the probability that a statement is true.
// A model with default instructions can distinguish an unset Payload from Null.
type Question struct {
	// Type selects "choice", "score" or "noul"; only h2o-lightning defaults to "choice" when it is omitted.
	Type string `json:"type,omitzero"`
	// Instructions supplies the question as text or JSON; omit it only for models with a default or fixed question.
	Instructions Payload `json:"instructions,omitzero"`
	// Criteria gives the options for choice or the ordered levels for score, and may be omitted for noul.
	Criteria Criteria `json:"criteria,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Choice asks which option best describes the state.
func Choice(instructions Payload, criteria ChoiceCriteria) Question {
	return Question{Type: "choice", Instructions: instructions, Criteria: criteria}
}

// Score asks which level describes the state.
func Score(instructions Payload, criteria ScoreCriteria) Question {
	return Question{Type: "score", Instructions: instructions, Criteria: criteria}
}

// Noul asks for the probability of a statement. A nil criteria argument omits the criteria field.
func Noul(instructions Payload, criteria *NoulCriteria) Question {
	q := Question{Type: "noul", Instructions: instructions}
	if criteria != nil {
		q.Criteria = criteria
	}
	return q
}

// MarshalJSONTo implements json.MarshalerTo.
func (q Question) MarshalJSONTo(enc *jsontext.Encoder) error {
	if err := reserved(q.Extra, "type", "instructions", "criteria"); err != nil {
		return err
	}
	var ok bool
	switch q.Type {
	case "choice", "":
		_, ok = q.Criteria.(ChoiceCriteria)
	case "score":
		_, ok = q.Criteria.(ScoreCriteria)
	case "noul":
		_, ok = q.Criteria.(*NoulCriteria)
		ok = ok || q.Criteria == nil
	default:
		return fmt.Errorf("systemone: unknown question type %q", q.Type)
	}
	if !ok {
		return fmt.Errorf("systemone: criteria %T do not match question type %q", q.Criteria, q.Type)
	}
	type wire Question
	return json.MarshalEncode(enc, wire(q))
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (q *Question) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var w struct {
		Type         string                    `json:"type"`
		Instructions Payload                   `json:"instructions"`
		Criteria     jsontext.Value            `json:"criteria"`
		Extra        map[string]jsontext.Value `json:",embed"`
	}
	if err := json.UnmarshalDecode(dec, &w); err != nil {
		return err
	}
	next := Question{Type: w.Type, Instructions: w.Instructions, Extra: w.Extra}
	switch w.Type {
	case "choice", "":
		var c ChoiceCriteria
		if w.Criteria.Kind() != '{' {
			return errors.New("systemone: choice criteria must be an object")
		}
		if err := json.Unmarshal(w.Criteria, &c); err != nil {
			return err
		}
		next.Criteria = c
	case "score":
		var c ScoreCriteria
		if w.Criteria.Kind() != '[' {
			return errors.New("systemone: score criteria must be an array")
		}
		if err := json.Unmarshal(w.Criteria, &c); err != nil {
			return err
		}
		next.Criteria = c
	case "noul":
		if w.Criteria != nil {
			var c *NoulCriteria
			if err := json.Unmarshal(w.Criteria, &c); err != nil {
				return err
			}
			next.Criteria = c
		}
	default:
		return fmt.Errorf("systemone: unknown question type %q", w.Type)
	}
	*q = next
	return nil
}

// MarshalJSONTo implements json.MarshalerTo.
func (c NoulCriteria) MarshalJSONTo(enc *jsontext.Encoder) error {
	if err := reserved(c.Extra, "false", "true", "no", "yes"); err != nil {
		return err
	}
	type wire NoulCriteria
	return json.MarshalEncode(enc, wire(c))
}
