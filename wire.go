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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
)

// jsonOptions are the options of every JSON encode and decode of the package. Deterministic output keeps the
// members of Extra maps in a stable order.
var jsonOptions = json.Deterministic(true)

// Token is a token id of the model's vocabulary.
type Token int32

// SamplingParams are the generation and sampling parameters that every completion-style endpoint accepts: the
// native /completion and /infill and the OpenAI-, Anthropic- and Responses-compatible endpoints, which pass them
// through to the same sampler.
//
// SamplingParams is meant to be embedded in request types, so that its members appear at the top level of the
// request object. It has no Extra field of its own because the request that embeds it collects unknown members.
// A nil pointer leaves the server default in place; new(expr) sets a value, including zero.
type SamplingParams struct {
	// NPredict caps generation in tokens; -1 is unlimited (the startup default) and 0 only evaluates the prompt.
	NPredict *int `json:"n_predict,omitzero"`
	// NKeep counts initial prompt tokens to retain on a context shift; 0 is the startup default and -1 keeps all.
	NKeep *int `json:"n_keep,omitzero"`
	// NDiscard counts tokens to drop after NKeep on a context shift; 0 or omission drops half the remaining tokens.
	NDiscard *int `json:"n_discard,omitzero"`
	// NIndent is the minimum generated line indentation in whitespace characters; defaults to 0.
	NIndent *int `json:"n_indent,omitzero"`
	// NCacheReuse is the minimum reusable chunk size in tokens for KV shifting; the startup default 0 disables it.
	NCacheReuse *int `json:"n_cache_reuse,omitzero"`
	// TMaxPredictMS caps elapsed milliseconds since the first generated token, but only after a newline appears;
	// the default 0 disables it.
	TMaxPredictMS *int `json:"t_max_predict_ms,omitzero"`
	// IDSlot assigns the task to a specific slot; -1 or omission lets the server choose.
	IDSlot *int `json:"id_slot,omitzero"`
	// CachePrompt reuses a matching prompt prefix from the cache when possible; enabled by default at startup.
	CachePrompt *bool `json:"cache_prompt,omitzero"`
	// ReturnTokens adds token IDs to buffered results; defaults to false and is automatic in streams.
	ReturnTokens *bool `json:"return_tokens,omitzero"`
	// ReturnProgress emits prompt processing progress events while streaming; defaults to false.
	ReturnProgress *bool `json:"return_progress,omitzero"`
	// TimingsPerToken includes timings in partial token events as well as the final result; defaults to false.
	TimingsPerToken *bool `json:"timings_per_token,omitzero"`
	// PostSamplingProbs reports probabilities after the sampler chain rather than log probabilities; default false.
	PostSamplingProbs *bool `json:"post_sampling_probs,omitzero"`
	// ResponseFields restricts the final native result to the listed fields, with no projection by default;
	// "a/b" selects a nested member and returns it under the literal top-level key "a/b".
	ResponseFields []string `json:"response_fields,omitzero"`
	// SSEPingInterval sets seconds between keep-alive comments in a silent stream; -1 disables them,
	// and nil uses the server's interval.
	SSEPingInterval *int `json:"sse_ping_interval,omitzero"`
	// Verbose requests the __verbose debug member in compatible responses; nil uses the server's verbosity setting.
	Verbose *bool `json:"verbose,omitzero"`

	// Temperature scales logits before sampling; 0 selects greedy decoding and the startup default is 0.8.
	Temperature *float64 `json:"temperature,omitzero"`
	// DynatempRange bounds temperature adjustment to Temperature ± this amount; the startup default 0 disables it.
	DynatempRange *float64 `json:"dynatemp_range,omitzero"`
	// DynatempExponent controls how entropy maps to dynamic temperature; the startup default is 1.
	DynatempExponent *float64 `json:"dynatemp_exponent,omitzero"`
	// TopK keeps the K most probable tokens; 0 disables it and the startup default is 40.
	TopK *int `json:"top_k,omitzero"`
	// TopP keeps tokens up to a cumulative probability in [0, 1]; 1 disables it and the startup default is 0.95.
	TopP *float64 `json:"top_p,omitzero"`
	// MinP drops tokens below this fraction of the top probability; defaults to 0.05 at startup (0 disables it).
	MinP *float64 `json:"min_p,omitzero"`
	// TopNSigma cuts off logits this many standard deviations below the top logit; the default -1 disables it.
	TopNSigma *float64 `json:"top_n_sigma,omitzero"`
	// XTCProbability is the chance that XTC removes high-probability candidates; the startup default 0 disables it.
	XTCProbability *float64 `json:"xtc_probability,omitzero"`
	// XTCThreshold is the minimum probability for XTC removal; above 0.5 disables it and the default is 0.1.
	XTCThreshold *float64 `json:"xtc_threshold,omitzero"`
	// TypicalP is the locally typical sampling cutoff; the startup default 1 disables it.
	TypicalP *float64 `json:"typical_p,omitzero"`
	// RepeatLastN is the recent-token window for repetition penalties; 0 disables it and the startup default is 64.
	RepeatLastN *int `json:"repeat_last_n,omitzero"`
	// RepeatPenalty downweights previously seen tokens; the startup default 1 disables it.
	RepeatPenalty *float64 `json:"repeat_penalty,omitzero"`
	// FrequencyPenalty is subtracted from a token's logit for each prior occurrence in the penalty window;
	// the startup default 0 disables it.
	FrequencyPenalty *float64 `json:"frequency_penalty,omitzero"`
	// PresencePenalty subtracts this amount from each previously seen token's logit; the startup default is 0.
	PresencePenalty *float64 `json:"presence_penalty,omitzero"`
	// DRYMultiplier scales the penalty for extending a repeated sequence; the startup default 0 disables DRY.
	DRYMultiplier *float64 `json:"dry_multiplier,omitzero"`
	// DRYBase is the exponential base for the DRY penalty; values below 1 use the server setting (1.75 at startup).
	DRYBase *float64 `json:"dry_base,omitzero"`
	// DRYAllowedLength is the repeated sequence length in tokens that goes unpenalized; the startup default is 2.
	DRYAllowedLength *int `json:"dry_allowed_length,omitzero"`
	// DRYPenaltyLastN is the recent-token window scanned for repeats; 0 disables it and the startup default is 64.
	DRYPenaltyLastN *int `json:"dry_penalty_last_n,omitzero"`
	// DRYSequenceBreakers supplies up to 64 distinct strings across which DRY does not count repeats;
	// the startup defaults are newline, colon, double quote, and asterisk.
	DRYSequenceBreakers []string `json:"dry_sequence_breakers,omitzero"`
	// Mirostat selects 0 for disabled (the startup default), 1 for Mirostat, or 2 for Mirostat 2.0.
	Mirostat *int `json:"mirostat,omitzero"`
	// MirostatTau is Mirostat's target token surprise in bits; the startup default is 5.
	MirostatTau *float64 `json:"mirostat_tau,omitzero"`
	// MirostatEta controls how quickly Mirostat adapts to its target; the startup default is 0.1.
	MirostatEta *float64 `json:"mirostat_eta,omitzero"`
	// AdaptiveTarget sets the adaptive-p probability target in [0, 1]; the startup default -1 disables it.
	AdaptiveTarget *float64 `json:"adaptive_target,omitzero"`
	// AdaptiveDecay controls how much adaptive-p weighs past tokens, in [0, 0.99]; the startup default is 0.9.
	AdaptiveDecay *float64 `json:"adaptive_decay,omitzero"`
	// Seed sets the random number generator seed; the startup default -1 picks a random seed.
	Seed *int64 `json:"seed,omitzero"`
	// NProbs caps candidate probabilities per generated token; the startup default 0 disables reporting.
	NProbs *int `json:"n_probs,omitzero"`
	// MinKeep sets the minimum candidate count for filtering samplers; the startup default 0 sets no minimum.
	MinKeep *int `json:"min_keep,omitzero"`
	// BackendSampling requests sampling on the model backend where supported; disabled by default at startup.
	BackendSampling *bool `json:"backend_sampling,omitzero"`
	// Samplers overrides the server's default chain with sampler names such as StringSlice("top_k", "top_p")
	// or a compact string such as StringValue("kpt").
	Samplers StringOrSlice `json:"samplers,omitzero"`
	// IgnoreEOS lets generation continue past an end-of-sequence token; disabled by default at startup.
	IgnoreEOS *bool `json:"ignore_eos,omitzero"`
	// LogitBias adjusts or bans token candidates before sampling; omission keeps the configured token biases.
	LogitBias LogitBias `json:"logit_bias,omitzero"`
	// Stop lists strings that stop generation without appearing in output; omitted, it uses the server's list.
	Stop StringOrSlice `json:"stop,omitzero"`

	// Grammar constrains output with GBNF; nil keeps the server's grammar unless JSONSchema is set.
	Grammar *string `json:"grammar,omitzero"`
	// JSONSchema is converted to a grammar unless Grammar is set; omission skips the conversion.
	JSONSchema jsontext.Value `json:"json_schema,omitzero"`
	// GrammarLazy delays grammar enforcement until a trigger matches; disabled by default at startup.
	GrammarLazy *bool `json:"grammar_lazy,omitzero"`
	// GrammarTriggers supplies words, tokens, or patterns that activate a lazy grammar; omission adds none.
	GrammarTriggers jsontext.Value `json:"grammar_triggers,omitzero"`
	// PreservedTokens supplies token strings that must not be split; omission adds none.
	PreservedTokens []string `json:"preserved_tokens,omitzero"`
	// LoRA overrides the server's adapter scales; with a nonempty override, unlisted adapters receive scale 0.
	LoRA []LoRAWeight `json:"lora,omitzero"`

	// ReasoningFormat selects how to parse reasoning, such as "none" or "deepseek"; nil uses the server setting.
	ReasoningFormat *string `json:"reasoning_format,omitzero"`
	// ReasoningControl lets a running request be told to end reasoning even without a token budget.
	ReasoningControl *bool `json:"reasoning_control,omitzero"`
	// ReasoningBudgetTokens caps the number of reasoning tokens; the startup default -1 disables the cap.
	ReasoningBudgetTokens *int `json:"reasoning_budget_tokens,omitzero"`
	// ReasoningBudgetStartTag marks where counting reasoning tokens starts; nil leaves the configured tag in place.
	ReasoningBudgetStartTag *string `json:"reasoning_budget_start_tag,omitzero"`
	// ReasoningBudgetEndTags supplies end markers, the first of which is forced when the budget expires;
	// omission leaves the configured markers in place.
	ReasoningBudgetEndTags []string `json:"reasoning_budget_end_tags,omitzero"`
	// ReasoningBudgetMessage supplies optional text to emit before the forced end marker.
	ReasoningBudgetMessage *string `json:"reasoning_budget_message,omitzero"`
}

// LoRAWeight sets the scale of one LoRA adapter.
type LoRAWeight struct {
	// ID is the adapter id reported by GET /lora-adapters.
	ID int `json:"id"`
	// Scale is the adapter scale; 0 disables it.
	Scale float64 `json:"scale"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Timings are the prompt processing and generation statistics of a completion.
type Timings struct {
	// CacheN is the number of prompt tokens reused from the cache.
	CacheN int `json:"cache_n"`
	// PromptN is the number of prompt tokens processed.
	PromptN int `json:"prompt_n"`
	// PromptMS is the prompt processing time in milliseconds.
	PromptMS float64 `json:"prompt_ms"`
	// PromptPerTokenMS is the prompt processing time per token in milliseconds.
	PromptPerTokenMS float64 `json:"prompt_per_token_ms"`
	// PromptPerSecond is the prompt processing speed in tokens per second.
	PromptPerSecond float64 `json:"prompt_per_second"`
	// PredictedN is the number of generated tokens.
	PredictedN int `json:"predicted_n"`
	// PredictedMS is the generation time in milliseconds.
	PredictedMS float64 `json:"predicted_ms"`
	// PredictedPerTokenMS is the generation time per token in milliseconds.
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	// PredictedPerSecond is the generation speed in tokens per second.
	PredictedPerSecond float64 `json:"predicted_per_second"`
	// DraftN is the number of draft tokens, present with speculative decoding.
	DraftN *int `json:"draft_n,omitzero"`
	// DraftNAccepted is the number of accepted draft tokens, present with speculative decoding.
	DraftNAccepted *int `json:"draft_n_accepted,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Usage is the token usage of an OpenAI-compatible chat or text completion.
type Usage struct {
	// CompletionTokens is the number of generated tokens.
	CompletionTokens int `json:"completion_tokens"`
	// PromptTokens is the number of prompt tokens.
	PromptTokens int `json:"prompt_tokens"`
	// TotalTokens is the sum of both.
	TotalTokens int `json:"total_tokens"`
	// PromptTokensDetails breaks down the prompt tokens.
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PromptTokensDetails breaks down the prompt tokens of a Usage.
type PromptTokensDetails struct {
	// CachedTokens is the number of prompt tokens reused from the cache.
	CachedTokens int `json:"cached_tokens"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PromptProgress reports prompt processing progress, streamed when ReturnProgress is set.
type PromptProgress struct {
	// Total is the number of prompt tokens.
	Total int `json:"total"`
	// Cache is the number of prompt tokens reused from the cache.
	Cache int `json:"cache"`
	// Processed is the number of prompt tokens processed so far.
	Processed int `json:"processed"`
	// TimeMS is the elapsed processing time in milliseconds.
	TimeMS float64 `json:"time_ms"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// TokenProb is the probability report of one generated token, used by the native completion_probabilities and by
// the OpenAI-compatible logprobs content.
//
// Without post_sampling_probs the server reports Logprob and TopLogprobs; with it, Prob and TopProbs.
type TokenProb struct {
	// ID is the token id.
	ID Token `json:"id"`
	// Token is the token text, cut to valid UTF-8.
	Token string `json:"token"`
	// Bytes are the raw bytes of the token text.
	Bytes []int `json:"bytes"`
	// Logprob is the log probability of the token.
	Logprob *float64 `json:"logprob,omitzero"`
	// Prob is the probability of the token after the sampler chain.
	Prob *float64 `json:"prob,omitzero"`
	// TopLogprobs are the most likely tokens with their log probabilities.
	TopLogprobs []TopTokenProb `json:"top_logprobs,omitzero"`
	// TopProbs are the most likely tokens with their probabilities after the sampler chain.
	TopProbs []TopTokenProb `json:"top_probs,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// TopTokenProb is one candidate token of a TokenProb.
type TopTokenProb struct {
	// ID is the token id.
	ID Token `json:"id"`
	// Token is the token text, cut to valid UTF-8.
	Token string `json:"token"`
	// Bytes are the raw bytes of the token text.
	Bytes []int `json:"bytes"`
	// Logprob is the log probability of the token.
	Logprob *float64 `json:"logprob,omitzero"`
	// Prob is the probability of the token after the sampler chain.
	Prob *float64 `json:"prob,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// PromptKind is the JSON shape of a Prompt.
type PromptKind int

const (
	// PromptKindNone is the zero Prompt, which is omitted from requests.
	PromptKindNone PromptKind = iota
	// PromptKindText is a string.
	PromptKindText
	// PromptKindTokens is an array of token ids.
	PromptKindTokens
	// PromptKindMixed is an array of strings and token ids.
	PromptKindMixed
	// PromptKindObject is an object with prompt_string and optional multimodal_data.
	PromptKindObject
	// PromptKindBatch is an array of prompts, each producing its own result.
	PromptKindBatch
)

// Prompt is the "prompt" (or "content") of a native request. The server accepts a string, an array of token ids,
// an array mixing strings and token ids, a PromptObject, or an array of those for several prompts at once.
// Build one with TextPrompt, TokenPrompt, MixedPrompt, ObjectPrompt, MultimodalPrompt or BatchPrompt.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type Prompt struct {
	kind   PromptKind
	text   string
	tokens []Token
	parts  []PromptPart
	object *PromptObject
	batch  []Prompt
}

// PromptObject is the object form of a Prompt.
type PromptObject struct {
	// PromptString is a single string (TokenizeText) or a flat array of text and token IDs (TokenizeParts).
	// With MultimodalData it must be a string containing the media markers.
	PromptString TokenizeContent `json:"prompt_string"`
	// MultimodalData are the media files (images, audio), sent base64-encoded.
	MultimodalData [][]byte `json:"multimodal_data,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

type promptObjectWire PromptObject

func (o *PromptObject) validate(hasMedia bool) error {
	if o.PromptString.IsZero() {
		return errors.New("llama: prompt_string is required")
	}
	if _, text := o.PromptString.Text(); hasMedia && !text {
		return errors.New("llama: prompt_string must be a string with multimodal_data")
	}
	return nil
}

// MarshalJSONTo implements json.MarshalerTo.
func (o PromptObject) MarshalJSONTo(enc *jsontext.Encoder) error {
	if err := o.validate(o.MultimodalData != nil); err != nil {
		return err
	}
	return json.MarshalEncode(enc, (*promptObjectWire)(&o))
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (o *PromptObject) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var raw struct {
		promptObjectWire
		Media jsontext.Value `json:"multimodal_data"`
	}
	if err := json.UnmarshalDecode(dec, &raw); err != nil {
		return err
	}
	next := PromptObject(raw.promptObjectWire)
	if err := next.validate(raw.Media != nil); err != nil {
		return err
	}
	if raw.Media != nil {
		if err := json.Unmarshal(raw.Media, &next.MultimodalData); err != nil {
			return err
		}
	}
	*o = next
	return nil
}

// PromptPart is one element of a mixed prompt: a string or a token id.
type PromptPart struct {
	text    string
	token   Token
	isToken bool
}

// TextPart returns a string element of a mixed prompt.
func TextPart(s string) PromptPart { return PromptPart{text: s} }

// TokenPart returns a token element of a mixed prompt.
func TokenPart(t Token) PromptPart { return PromptPart{token: t, isToken: true} }

// Text returns the string of a string element.
func (p PromptPart) Text() (string, bool) { return p.text, !p.isToken }

// Token returns the token id of a token element.
func (p PromptPart) Token() (Token, bool) { return p.token, p.isToken }

// TextPrompt returns a prompt given as a string.
func TextPrompt(s string) Prompt { return Prompt{kind: PromptKindText, text: s} }

// TokenPrompt returns a prompt given as token ids.
func TokenPrompt(tokens ...Token) Prompt {
	return Prompt{kind: PromptKindTokens, tokens: slices.Clone(tokens)}
}

// MixedPrompt returns a prompt mixing strings and token ids. It must contain at least one token: the server reads
// an array without token ids as several prompts.
func MixedPrompt(parts ...PromptPart) Prompt {
	return Prompt{kind: PromptKindMixed, parts: slices.Clone(parts)}
}

// ObjectPrompt returns a prompt in object form.
func ObjectPrompt(o PromptObject) Prompt { return Prompt{kind: PromptKindObject, object: &o} }

// MultimodalPrompt returns a prompt carrying media files. text holds one media marker per file.
func MultimodalPrompt(text string, data ...[]byte) Prompt {
	return ObjectPrompt(PromptObject{PromptString: TokenizeText(text), MultimodalData: data})
}

// BatchPrompt returns several prompts sent in one request. The elements must not be batches themselves.
func BatchPrompt(prompts ...Prompt) Prompt {
	return Prompt{kind: PromptKindBatch, batch: slices.Clone(prompts)}
}

// Kind returns the shape of the prompt.
func (p Prompt) Kind() PromptKind { return p.kind }

// Text returns the string of a PromptKindText prompt.
func (p Prompt) Text() string { return p.text }

// Tokens returns a read-only view of the token ids of a PromptKindTokens prompt.
func (p Prompt) Tokens() []Token { return p.tokens }

// Parts returns a read-only view of the elements of a PromptKindMixed prompt.
func (p Prompt) Parts() []PromptPart { return p.parts }

// Object returns the object of a PromptKindObject prompt.
func (p Prompt) Object() *PromptObject { return p.object }

// Batch returns a read-only view of the prompts of a PromptKindBatch prompt.
func (p Prompt) Batch() []Prompt { return p.batch }

// IsZero reports whether the prompt is unset.
func (p Prompt) IsZero() bool { return p.kind == PromptKindNone }

// Equal reports whether two prompts have the same shape and content.
func (p Prompt) Equal(q Prompt) bool {
	if p.kind != q.kind {
		return false
	}
	switch p.kind {
	case PromptKindText:
		return p.text == q.text
	case PromptKindTokens:
		return slices.Equal(p.tokens, q.tokens)
	case PromptKindMixed:
		return slices.Equal(p.parts, q.parts)
	case PromptKindObject:
		return p.object.equal(q.object)
	case PromptKindBatch:
		return slices.EqualFunc(p.batch, q.batch, Prompt.Equal)
	default:
		return true
	}
}

func (o *PromptObject) equal(q *PromptObject) bool {
	if o == nil || q == nil {
		return o == q
	}
	ot, os := o.PromptString.Text()
	qt, qs := q.PromptString.Text()
	return ot == qt && os == qs && o.PromptString.IsZero() == q.PromptString.IsZero() &&
		slices.Equal(o.PromptString.Parts(), q.PromptString.Parts()) &&
		slices.EqualFunc(o.MultimodalData, q.MultimodalData, bytes.Equal) &&
		maps.EqualFunc(o.Extra, q.Extra, func(a, b jsontext.Value) bool { return bytes.Equal(a, b) })
}

// MarshalJSONTo implements json.MarshalerTo.
func (p Prompt) MarshalJSONTo(enc *jsontext.Encoder) error {
	if p.kind == PromptKindBatch {
		if err := enc.WriteToken(jsontext.BeginArray); err != nil {
			return err
		}
		for _, sub := range p.batch {
			if sub.kind == PromptKindBatch || sub.kind == PromptKindNone {
				return errors.New("llama: a batch prompt element must be a text, token, mixed or object prompt")
			}
			if err := sub.marshalSingle(enc); err != nil {
				return err
			}
		}
		return enc.WriteToken(jsontext.EndArray)
	}
	return p.marshalSingle(enc)
}

func (p Prompt) marshalSingle(enc *jsontext.Encoder) error {
	switch p.kind {
	case PromptKindNone:
		return enc.WriteToken(jsontext.Null)
	case PromptKindText:
		return enc.WriteToken(jsontext.String(p.text))
	case PromptKindTokens:
		return json.MarshalEncode(enc, p.tokens)
	case PromptKindMixed:
		if !slices.ContainsFunc(p.parts, func(pp PromptPart) bool { return pp.isToken }) {
			return errors.New("llama: a mixed prompt needs at least one token; the server reads an array of strings as several prompts")
		}
		if err := enc.WriteToken(jsontext.BeginArray); err != nil {
			return err
		}
		for _, pp := range p.parts {
			tok := jsontext.String(pp.text)
			if pp.isToken {
				tok = jsontext.Int(int64(pp.token))
			}
			if err := enc.WriteToken(tok); err != nil {
				return err
			}
		}
		return enc.WriteToken(jsontext.EndArray)
	case PromptKindObject:
		return json.MarshalEncode(enc, p.object)
	default:
		return fmt.Errorf("llama: invalid prompt kind %d", p.kind)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom with the server's rules: a top-level array that contains a
// token id is one prompt, any other top-level array is a batch.
func (p *Prompt) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
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
			batch := make([]Prompt, len(elems))
			for i, e := range elems {
				// The server rejects a null element, and a batch with an unset element cannot be encoded.
				if e.Kind() == 'n' {
					return fmt.Errorf("llama: prompt %d of a batch is null", i)
				}
				if err := batch[i].unmarshalSingle(e); err != nil {
					return fmt.Errorf("prompt %d: %w", i, err)
				}
			}
			*p = Prompt{kind: PromptKindBatch, batch: batch}
			return nil
		}
	}
	return p.unmarshalSingle(v)
}

func isNumber(v jsontext.Value) bool { return v.Kind() == '0' }

func (p *Prompt) unmarshalSingle(v jsontext.Value) error {
	switch v.Kind() {
	case 'n':
		*p = Prompt{}
		return nil
	case '"':
		var s string
		if err := json.Unmarshal(v, &s); err != nil {
			return err
		}
		*p = TextPrompt(s)
		return nil
	case '{':
		var o PromptObject
		if err := json.Unmarshal(v, &o, jsonOptions); err != nil {
			return err
		}
		*p = Prompt{kind: PromptKindObject, object: &o}
		return nil
	case '[':
		var elems []jsontext.Value
		if err := json.Unmarshal(v, &elems); err != nil {
			return err
		}
		if !slices.ContainsFunc(elems, isNumber) {
			return errors.New("llama: a prompt array must contain at least one token id")
		}
		parts := make([]PromptPart, len(elems))
		allTokens := true
		for i, e := range elems {
			switch e.Kind() {
			case '0':
				var t Token
				if err := json.Unmarshal(e, &t); err != nil {
					return fmt.Errorf("llama: prompt element %d: %w", i, err)
				}
				parts[i] = TokenPart(t)
			case '"':
				var s string
				if err := json.Unmarshal(e, &s); err != nil {
					return err
				}
				parts[i] = TextPart(s)
				allTokens = false
			default:
				return fmt.Errorf("llama: prompt element %d must be a string or a token id, got %s", i, e.Kind())
			}
		}
		if allTokens {
			tokens := make([]Token, len(parts))
			for i, pp := range parts {
				tokens[i] = pp.token
			}
			*p = Prompt{kind: PromptKindTokens, tokens: tokens}
			return nil
		}
		*p = Prompt{kind: PromptKindMixed, parts: parts}
		return nil
	default:
		return fmt.Errorf("llama: a prompt must be a string, an array or an object, got %s", v.Kind())
	}
}

// LogitBiasTarget is what a LogitBiasEntry applies to: a token id, or a string whose tokens all receive the bias.
// Build one with TokenTarget or TextTarget.
type LogitBiasTarget struct {
	text   string
	token  Token
	isText bool
}

// TokenTarget returns a target naming a token id.
func TokenTarget(t Token) LogitBiasTarget { return LogitBiasTarget{token: t} }

// TextTarget returns a target naming a string, which the server tokenizes. In object form the server reads a key
// that is entirely a decimal integer (for example "01") as a token id, so such a text reaches that token.
func TextTarget(s string) LogitBiasTarget { return LogitBiasTarget{text: s, isText: true} }

// Token returns the token id of a token target.
func (t LogitBiasTarget) Token() (Token, bool) { return t.token, !t.isText }

// Text returns the string of a text target.
func (t LogitBiasTarget) Text() (string, bool) { return t.text, t.isText }

func (t LogitBiasTarget) key() string {
	if t.isText {
		return t.text
	}
	return strconv.FormatInt(int64(t.token), 10)
}

// targetFromKey returns the target of an object member name. Only the canonical spelling of an integer becomes a
// token target, so that every other spelling ("01", "+1") is kept as written.
func targetFromKey(key string) LogitBiasTarget {
	if n, err := strconv.ParseInt(key, 10, 32); err == nil && strconv.FormatInt(n, 10) == key {
		return TokenTarget(Token(n))
	}
	return TextTarget(key)
}

// LogitBiasEntry changes the likelihood of a target.
type LogitBiasEntry struct {
	// Target is the token or string the entry applies to.
	Target LogitBiasTarget
	// Bias is added to the logit.
	Bias float64
	// Ban forbids the target; it is sent as false and Bias is ignored.
	Ban bool
}

// LogitBias is the "logit_bias" of a request. The server accepts an array of [token or string, bias] pairs or an
// object mapping a token id or string to a bias, where a bias of false bans the token. Build one with
// LogitBiasArray or LogitBiasObject.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type LogitBias struct {
	entries []LogitBiasEntry
	object  bool
}

// LogitBiasArray returns a logit bias sent in array form.
func LogitBiasArray(entries ...LogitBiasEntry) LogitBias {
	return LogitBias{entries: slices.Clone(entries)}
}

// LogitBiasObject returns a logit bias sent in object form, the form OpenAI clients use. Two entries whose targets
// spell the same member name (TokenTarget(1) and TextTarget("1")) cannot be encoded.
func LogitBiasObject(entries ...LogitBiasEntry) LogitBias {
	return LogitBias{entries: slices.Clone(entries), object: true}
}

// Entries returns a read-only view of the entries in order.
func (b LogitBias) Entries() []LogitBiasEntry { return b.entries }

// IsObject reports whether the bias is sent in object form.
func (b LogitBias) IsObject() bool { return b.object }

// IsZero reports whether the bias is unset.
func (b LogitBias) IsZero() bool { return b.entries == nil && !b.object }

// Equal reports whether two biases have the same form and entries.
func (b LogitBias) Equal(o LogitBias) bool {
	return b.object == o.object && slices.Equal(b.entries, o.entries)
}

func (e LogitBiasEntry) biasToken() jsontext.Token {
	if e.Ban {
		return jsontext.False
	}
	return jsontext.Float(e.Bias)
}

// MarshalJSONTo implements json.MarshalerTo.
func (b LogitBias) MarshalJSONTo(enc *jsontext.Encoder) error {
	if b.object {
		if err := enc.WriteToken(jsontext.BeginObject); err != nil {
			return err
		}
		for _, e := range b.entries {
			if err := enc.WriteToken(jsontext.String(e.Target.key())); err != nil {
				return err
			}
			if err := enc.WriteToken(e.biasToken()); err != nil {
				return err
			}
		}
		return enc.WriteToken(jsontext.EndObject)
	}
	if err := enc.WriteToken(jsontext.BeginArray); err != nil {
		return err
	}
	for _, e := range b.entries {
		target := jsontext.Int(int64(e.Target.token))
		if e.Target.isText {
			target = jsontext.String(e.Target.text)
		}
		for _, tok := range []jsontext.Token{jsontext.BeginArray, target, e.biasToken(), jsontext.EndArray} {
			if err := enc.WriteToken(tok); err != nil {
				return err
			}
		}
	}
	return enc.WriteToken(jsontext.EndArray)
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (b *LogitBias) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '{':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		// The members are read one by one because their order is part of the value.
		entries := []LogitBiasEntry{}
		for dec.PeekKind() != '}' {
			k, err := dec.ReadToken()
			if err != nil {
				return err
			}
			// A Token is only valid until the next decoder call.
			key := k.String()
			val, err := dec.ReadValue()
			if err != nil {
				return err
			}
			e := LogitBiasEntry{Target: targetFromKey(key)}
			if err := e.unmarshalBias(val); err != nil {
				return fmt.Errorf("llama: logit_bias %q: %w", key, err)
			}
			entries = append(entries, e)
		}
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*b = LogitBias{entries: entries, object: true}
		return nil
	case '[':
		var pairs [][]jsontext.Value
		if err := json.UnmarshalDecode(dec, &pairs); err != nil {
			return fmt.Errorf("llama: logit_bias array must hold [token, bias] pairs: %w", err)
		}
		entries := make([]LogitBiasEntry, 0, len(pairs))
		for i, pair := range pairs {
			if len(pair) != 2 {
				return fmt.Errorf("llama: logit_bias pair %d has %d elements, want 2", i, len(pair))
			}
			var e LogitBiasEntry
			switch pair[0].Kind() {
			case '0':
				var t Token
				if err := json.Unmarshal(pair[0], &t); err != nil {
					return fmt.Errorf("llama: logit_bias pair %d: %w", i, err)
				}
				e.Target = TokenTarget(t)
			case '"':
				var text string
				if err := json.Unmarshal(pair[0], &text); err != nil {
					return err
				}
				e.Target = TextTarget(text)
			default:
				return fmt.Errorf("llama: logit_bias pair %d target must be a token id or a string", i)
			}
			if err := e.unmarshalBias(pair[1]); err != nil {
				return fmt.Errorf("llama: logit_bias pair %d: %w", i, err)
			}
			entries = append(entries, e)
		}
		*b = LogitBias{entries: entries}
		return nil
	case 'n':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*b = LogitBias{}
		return nil
	default:
		return fmt.Errorf("llama: logit_bias must be an array or an object, got %s", kind)
	}
}

func (e *LogitBiasEntry) unmarshalBias(v jsontext.Value) error {
	switch v.Kind() {
	case '0':
		return json.Unmarshal(v, &e.Bias)
	case 'f':
		e.Ban = true
		return nil
	default:
		return fmt.Errorf("bias must be a number or false, got %s", v.Kind())
	}
}

// StringOrSlice is a JSON value that is either a string or an array of strings, such as "stop". It remembers which
// form it was built or decoded from, so that it encodes back to the same form. Build one with StringValue or
// StringSlice.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type StringOrSlice struct {
	values []string
	single bool
}

// StringValue returns a StringOrSlice encoded as a single string.
func StringValue(s string) StringOrSlice { return StringOrSlice{values: []string{s}, single: true} }

// StringSlice returns a StringOrSlice encoded as an array of strings.
func StringSlice(s ...string) StringOrSlice {
	if s == nil {
		s = []string{}
	}
	return StringOrSlice{values: slices.Clone(s)}
}

// Strings returns a read-only view of the strings; a single string is returned as a one-element slice.
func (s StringOrSlice) Strings() []string { return s.values }

// IsSingle reports whether the value is encoded as a single string.
func (s StringOrSlice) IsSingle() bool { return s.single }

// IsZero reports whether the value is unset.
func (s StringOrSlice) IsZero() bool { return s.values == nil }

// Equal reports whether two values have the same form and strings.
func (s StringOrSlice) Equal(o StringOrSlice) bool {
	return s.single == o.single && slices.Equal(s.values, o.values) && (s.values == nil) == (o.values == nil)
}

// MarshalJSONTo implements json.MarshalerTo.
func (s StringOrSlice) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case s.values == nil:
		return enc.WriteToken(jsontext.Null)
	case s.single:
		return enc.WriteToken(jsontext.String(s.values[0]))
	default:
		return json.MarshalEncode(enc, s.values)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (s *StringOrSlice) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch dec.PeekKind() {
	case '"':
		var v string
		if err := json.UnmarshalDecode(dec, &v); err != nil {
			return err
		}
		*s = StringValue(v)
		return nil
	case '[':
		var v []string
		if err := json.UnmarshalDecode(dec, &v); err != nil {
			return err
		}
		*s = StringSlice(v...)
		return nil
	case 'n':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
		*s = StringOrSlice{}
		return nil
	default:
		return fmt.Errorf("llama: expected a string or an array of strings, got %s", dec.PeekKind())
	}
}
