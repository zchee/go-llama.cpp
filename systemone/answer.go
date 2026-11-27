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

// ChoiceAnswer names the most probable option and gives the distribution across all options.
type ChoiceAnswer struct {
	// Choice is the key of the highest-probability option in the request criteria.
	Choice string `json:"choice"`
	// Probabilities maps every option key to its probability, with a total of 1.
	Probabilities map[string]float64 `json:"probabilities"`
	// Confidence measures how concentrated the option probabilities are on a model-dependent scale from 0 to 1.
	Confidence float64 `json:"confidence"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ScoreAnswer reports the expected level index (the median level for lfm3-d1 models).
type ScoreAnswer struct {
	// Score is the probability-weighted level index starting at 0, or the median level for lfm3-d1 models.
	Score float64 `json:"score"`
	// Legend maps stringified level indexes to their request descriptions.
	Legend map[string]jsontext.Value `json:"legend"`
	// Probabilities maps stringified level indexes to probabilities totaling 1.
	Probabilities map[string]float64 `json:"probabilities"`
	// Confidence is a model-dependent certainty measure from 0 to 1.
	Confidence float64 `json:"confidence"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// NoulAnswer gives the probability that a statement is true.
type NoulAnswer struct {
	// Noul is the model's estimated probability from 0 to 1 that the statement is true.
	Noul float64 `json:"noul"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AnswerValue is a ChoiceAnswer, ScoreAnswer or NoulAnswer, as selected by the wire type.
type AnswerValue interface{ isAnswer() }

func (ChoiceAnswer) isAnswer() {}
func (ScoreAnswer) isAnswer()  {}
func (NoulAnswer) isAnswer()   {}

// Answer contains a typed result. Use a type switch on Value to inspect it.
type Answer struct {
	// Value is the decoded result selected by the server's type discriminator.
	Value AnswerValue
}

type choiceWire struct {
	Type string `json:"type"`
	ChoiceAnswer
}
type scoreWire struct {
	Type string `json:"type"`
	ScoreAnswer
}
type noulWire struct {
	Type string `json:"type"`
	NoulAnswer
}

// MarshalJSONTo implements json.MarshalerTo.
func (a Answer) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch v := a.Value.(type) {
	case ChoiceAnswer:
		return json.MarshalEncode(enc, choiceWire{"choice", v})
	case ScoreAnswer:
		return json.MarshalEncode(enc, scoreWire{"score", v})
	case NoulAnswer:
		return json.MarshalEncode(enc, noulWire{"noul", v})
	default:
		return fmt.Errorf("systemone: unsupported answer %T", a.Value)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (a *Answer) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
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
	switch tag.Type {
	case "choice":
		var v choiceWire
		if err := json.Unmarshal(raw, &v); err != nil {
			return err
		}
		if v.Probabilities == nil {
			return errors.New("systemone: choice answer requires probabilities")
		}
		a.Value = v.ChoiceAnswer
	case "score":
		var v scoreWire
		if err := json.Unmarshal(raw, &v); err != nil {
			return err
		}
		if v.Probabilities == nil || v.Legend == nil {
			return errors.New("systemone: score answer requires probabilities and a legend")
		}
		a.Value = v.ScoreAnswer
	case "noul":
		var v noulWire
		if err := json.Unmarshal(raw, &v); err != nil {
			return err
		}
		a.Value = v.NoulAnswer
	default:
		return fmt.Errorf("systemone: unknown answer type %q", tag.Type)
	}
	return nil
}
