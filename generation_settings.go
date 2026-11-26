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
	// Seed is the configured random number generator seed; 4294967295 asks the server to choose it randomly.
	Seed uint32 `json:"seed"`
	// Temperature scales the logits before sampling; 0 means greedy decoding.
	Temperature float64 `json:"temperature"`
	// DynatempRange is the maximum adjustment above or below Temperature; 0 disables dynamic temperature.
	DynatempRange float64 `json:"dynatemp_range"`
	// DynatempExponent controls how token-distribution entropy maps to dynamic temperature.
	DynatempExponent float64 `json:"dynatemp_exponent"`
	// TopK caps sampling at the K most probable tokens; 0 disables the cap.
	TopK int `json:"top_k"`
	// TopP is the cumulative probability cutoff for nucleus sampling; 1 disables it.
	TopP float64 `json:"top_p"`
	// MinP is the minimum probability relative to the most likely token; 0 disables the cutoff.
	MinP float64 `json:"min_p"`
	// TopNSigma is the cutoff in standard deviations below the highest logit; a negative value disables it.
	TopNSigma float64 `json:"top_n_sigma"`
	// XTCProbability is the chance the XTC sampler removes high-probability candidates; 0 disables it.
	XTCProbability float64 `json:"xtc_probability"`
	// XTCThreshold is the minimum probability for XTC removal; values above 0.5 disable it.
	XTCThreshold float64 `json:"xtc_threshold"`
	// TypicalP is the locally typical sampling cutoff; 1 disables it.
	TypicalP float64 `json:"typical_p"`
	// RepeatLastN is the number of recent tokens considered by the repetition penalties; 0 disables them.
	RepeatLastN int `json:"repeat_last_n"`
	// RepeatPenalty downweights previously seen tokens; 1 disables it.
	RepeatPenalty float64 `json:"repeat_penalty"`
	// PresencePenalty is subtracted once from the logit of each previously seen token; 0 disables it.
	PresencePenalty float64 `json:"presence_penalty"`
	// FrequencyPenalty is subtracted from a token's logit for each occurrence in the penalty window; 0 disables it.
	FrequencyPenalty float64 `json:"frequency_penalty"`
	// DRYMultiplier scales the penalty for extending a repeated sequence; 0 disables DRY.
	DRYMultiplier float64 `json:"dry_multiplier"`
	// DRYBase is the exponential base for the DRY penalty beyond DRYAllowedLength.
	DRYBase float64 `json:"dry_base"`
	// DRYAllowedLength is the maximum repeated sequence length, in tokens, that DRY leaves unpenalized.
	DRYAllowedLength int `json:"dry_allowed_length"`
	// DRYPenaltyLastN is the number of recent tokens scanned by DRY; 0 disables the scan.
	DRYPenaltyLastN int `json:"dry_penalty_last_n"`
	// DRYSequenceBreakers are strings across which DRY does not count repeats; omitted in reduced settings.
	DRYSequenceBreakers []string `json:"dry_sequence_breakers,omitzero"`
	// Mirostat identifies the sampler variant: 0 for disabled, 1 for Mirostat, or 2 for Mirostat 2.0.
	Mirostat int `json:"mirostat"`
	// MirostatTau is the target token surprise in bits for Mirostat.
	MirostatTau float64 `json:"mirostat_tau"`
	// MirostatEta determines how quickly Mirostat adjusts to its target surprise.
	MirostatEta float64 `json:"mirostat_eta"`
	// AdaptiveTarget is the target token probability in [0, 1] for adaptive-p; a negative value disables it.
	AdaptiveTarget float64 `json:"adaptive_target"`
	// AdaptiveDecay is the adaptive-p history decay in [0, 0.99]; higher values weigh past tokens more heavily.
	AdaptiveDecay float64 `json:"adaptive_decay"`
	// Stop lists strings that end generation without appearing in the output; omitted in reduced settings.
	Stop []string `json:"stop,omitzero"`
	// MaxTokens repeats NPredict under the max_tokens wire name.
	MaxTokens int `json:"max_tokens"`
	// NPredict is the configured generation token limit; -1 means no limit and 0 means prompt processing only.
	NPredict int `json:"n_predict"`
	// NKeep counts initial prompt tokens to retain when the context shifts; -1 keeps the full prompt.
	NKeep int `json:"n_keep"`
	// NDiscard counts tokens to drop after NKeep on a context shift; 0 means half of the remaining tokens.
	NDiscard int `json:"n_discard"`
	// IgnoreEOS indicates that the end-of-sequence token does not stop generation.
	IgnoreEOS bool `json:"ignore_eos"`
	// Stream indicates that output was requested as server-sent events rather than buffered JSON.
	Stream bool `json:"stream"`
	// LogitBias lists resolved per-token logit adjustments and bans; omitted in reduced settings.
	LogitBias []GenerationSettingsBias `json:"logit_bias,omitzero"`
	// NProbs caps the number of alternative token probabilities reported per generated token; 0 disables reporting.
	NProbs int `json:"n_probs"`
	// MinKeep is the minimum number of candidates that filtering samplers should retain; 0 sets no minimum.
	MinKeep int `json:"min_keep"`
	// Grammar is the effective GBNF, including grammar derived from JSON Schema; omitted in reduced settings.
	Grammar *string `json:"grammar,omitzero"`
	// GrammarLazy indicates whether the grammar waits for a trigger; omitted in reduced settings.
	GrammarLazy *bool `json:"grammar_lazy,omitzero"`
	// GrammarTriggers lists the conditions that activate a lazy grammar; omitted in reduced settings.
	GrammarTriggers []GenerationSettingsTrigger `json:"grammar_triggers,omitzero"`
	// PreservedTokens lists vocabulary IDs resolved from the preserved token strings; omitted in reduced settings.
	PreservedTokens []Token `json:"preserved_tokens,omitzero"`
	// ChatFormat identifies the parser used to interpret the model's generated chat text.
	ChatFormat string `json:"chat_format"`
	// ReasoningFormat identifies how the server separates reasoning from the model's answer.
	ReasoningFormat string `json:"reasoning_format"`
	// ReasoningInContent is true for DeepSeek legacy streams, which leave reasoning in the content.
	ReasoningInContent bool `json:"reasoning_in_content"`
	// GenerationPrompt is the template-rendered start of the assistant turn used by the sampler.
	GenerationPrompt string `json:"generation_prompt"`
	// Samplers lists sampler names in the order configured for the chain.
	Samplers []string `json:"samplers"`
	// SpeculativeTypes reports the configured speculative decoding methods under the literal key speculative.types.
	SpeculativeTypes string `json:"speculative.types"`
	// TimingsPerToken indicates that timings were requested in the partial stream events.
	TimingsPerToken bool `json:"timings_per_token"`
	// PostSamplingProbs indicates that reported probabilities are taken after the sampler chain.
	PostSamplingProbs bool `json:"post_sampling_probs"`
	// BackendSampling indicates that sampling on the model backend was requested, if supported for the task.
	BackendSampling bool `json:"backend_sampling"`
	// LoRA lists the adapter IDs and scales selected for generation.
	LoRA []LoRAWeight `json:"lora"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// GenerationSettingsBias is a resolved token bias; the server serializes an infinite bias as null.
type GenerationSettingsBias struct {
	// Token is the vocabulary ID whose logit is changed.
	Token Token `json:"token"`
	// Bias is added to the token's logit before sampling; nil represents an infinite penalty that bans it.
	Bias *float64 `json:"bias"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// GenerationSettingsTrigger describes a lazy grammar trigger; Token is present only for a token trigger.
type GenerationSettingsTrigger struct {
	// Type identifies the condition as a word, token, or regular expression.
	Type int `json:"type"`
	// Value is the word or pattern that activates the grammar, or the source spelling of a token trigger.
	Value string `json:"value"`
	// Token is the vocabulary ID that activates the grammar; omitted for word and pattern triggers.
	Token *Token `json:"token,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}
