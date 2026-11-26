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

import "encoding/json/jsontext"

// GenerationSettings reports the effective generation parameters, rather than the request that selected them.
// Properties and slot metrics omit some of the more expensive fields, including grammar and logit biases.
type GenerationSettings struct {
	Seed                uint32                      `json:"seed"`
	Temperature         float64                     `json:"temperature"`
	DynatempRange       float64                     `json:"dynatemp_range"`
	DynatempExponent    float64                     `json:"dynatemp_exponent"`
	TopK                int                         `json:"top_k"`
	TopP                float64                     `json:"top_p"`
	MinP                float64                     `json:"min_p"`
	TopNSigma           float64                     `json:"top_n_sigma"`
	XTCProbability      float64                     `json:"xtc_probability"`
	XTCThreshold        float64                     `json:"xtc_threshold"`
	TypicalP            float64                     `json:"typical_p"`
	RepeatLastN         int                         `json:"repeat_last_n"`
	RepeatPenalty       float64                     `json:"repeat_penalty"`
	PresencePenalty     float64                     `json:"presence_penalty"`
	FrequencyPenalty    float64                     `json:"frequency_penalty"`
	DRYMultiplier       float64                     `json:"dry_multiplier"`
	DRYBase             float64                     `json:"dry_base"`
	DRYAllowedLength    int                         `json:"dry_allowed_length"`
	DRYPenaltyLastN     int                         `json:"dry_penalty_last_n"`
	DRYSequenceBreakers []string                    `json:"dry_sequence_breakers,omitzero"`
	Mirostat            int                         `json:"mirostat"`
	MirostatTau         float64                     `json:"mirostat_tau"`
	MirostatEta         float64                     `json:"mirostat_eta"`
	AdaptiveTarget      float64                     `json:"adaptive_target"`
	AdaptiveDecay       float64                     `json:"adaptive_decay"`
	Stop                []string                    `json:"stop,omitzero"`
	MaxTokens           int                         `json:"max_tokens"`
	NPredict            int                         `json:"n_predict"`
	NKeep               int                         `json:"n_keep"`
	NDiscard            int                         `json:"n_discard"`
	IgnoreEOS           bool                        `json:"ignore_eos"`
	Stream              bool                        `json:"stream"`
	LogitBias           []GenerationSettingsBias    `json:"logit_bias,omitzero"`
	NProbs              int                         `json:"n_probs"`
	MinKeep             int                         `json:"min_keep"`
	Grammar             *string                     `json:"grammar,omitzero"`
	GrammarLazy         *bool                       `json:"grammar_lazy,omitzero"`
	GrammarTriggers     []GenerationSettingsTrigger `json:"grammar_triggers,omitzero"`
	PreservedTokens     []Token                     `json:"preserved_tokens,omitzero"`
	ChatFormat          string                      `json:"chat_format"`
	ReasoningFormat     string                      `json:"reasoning_format"`
	ReasoningInContent  bool                        `json:"reasoning_in_content"`
	GenerationPrompt    string                      `json:"generation_prompt"`
	Samplers            []string                    `json:"samplers"`
	SpeculativeTypes    string                      `json:"speculative.types"`
	TimingsPerToken     bool                        `json:"timings_per_token"`
	PostSamplingProbs   bool                        `json:"post_sampling_probs"`
	BackendSampling     bool                        `json:"backend_sampling"`
	LoRA                []LoRAWeight                `json:"lora"`
	Extra               map[string]jsontext.Value   `json:",embed"`
}

// GenerationSettingsBias is a resolved token bias; the server serializes an infinite bias as null.
type GenerationSettingsBias struct {
	Token Token                     `json:"token"`
	Bias  *float64                  `json:"bias"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// GenerationSettingsTrigger describes a lazy grammar trigger; Token is present only for a token trigger.
type GenerationSettingsTrigger struct {
	Type  int                       `json:"type"`
	Value string                    `json:"value"`
	Token *Token                    `json:"token,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}
