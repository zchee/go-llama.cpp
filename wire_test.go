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
	"reflect"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestPromptJSON(t *testing.T) {
	tests := map[string]struct {
		prompt   Prompt
		json     string
		wantKind PromptKind
	}{
		"success: text":                   {prompt: TextPrompt("Hello"), json: `"Hello"`, wantKind: PromptKindText},
		"success: empty text is not zero": {prompt: TextPrompt(""), json: `""`, wantKind: PromptKindText},
		"success: tokens":                 {prompt: TokenPrompt(12, 34, 56), json: `[12,34,56]`, wantKind: PromptKindTokens},
		"success: mixed": {
			prompt:   MixedPrompt(TokenPart(12), TokenPart(34), TextPart("string"), TokenPart(56)),
			json:     `[12,34,"string",56]`,
			wantKind: PromptKindMixed,
		},
		"success: object": {
			prompt:   MultimodalPrompt("<__media__> describe", []byte("\x89PNG")),
			json:     `{"prompt_string":"<__media__> describe","multimodal_data":["iVBORw=="]}`,
			wantKind: PromptKindObject,
		},
		"success: all strings in an object are one prompt": {
			prompt:   ObjectPrompt(PromptObject{PromptString: TokenizeParts(TextPart("a"), TextPart("b"))}),
			json:     `{"prompt_string":["a","b"]}`,
			wantKind: PromptKindObject,
		},
		"success: object keeps unknown members": {
			prompt:   ObjectPrompt(PromptObject{PromptString: TokenizeText("x"), Extra: map[string]jsontext.Value{"future": jsontext.Value(`{"a":1}`)}}),
			json:     `{"prompt_string":"x","future":{"a":1}}`,
			wantKind: PromptKindObject,
		},
		"success: batch of every element kind": {
			prompt: BatchPrompt(
				TextPrompt("a"),
				TokenPrompt(1, 2),
				MixedPrompt(TextPart("b"), TokenPart(3)),
				ObjectPrompt(PromptObject{PromptString: TokenizeText("c")}),
			),
			json:     `["a",[1,2],["b",3],{"prompt_string":"c"}]`,
			wantKind: PromptKindBatch,
		},
		"success: batch of strings": {
			prompt:   BatchPrompt(TextPrompt("a"), TextPrompt("b")),
			json:     `["a","b"]`,
			wantKind: PromptKindBatch,
		},
		"success: empty batch": {prompt: BatchPrompt(), json: `[]`, wantKind: PromptKindBatch},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := json.Marshal(tt.prompt, jsonOptions)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if string(got) != tt.json {
				t.Errorf("Marshal = %s, want %s", got, tt.json)
			}
			var back Prompt
			if err := json.Unmarshal([]byte(tt.json), &back, jsonOptions); err != nil {
				t.Fatalf("Unmarshal(%s): %v", tt.json, err)
			}
			if back.Kind() != tt.wantKind {
				t.Errorf("Kind = %d, want %d", back.Kind(), tt.wantKind)
			}
			if diff := gocmp.Diff(tt.prompt, back); diff != "" {
				t.Errorf("decoded prompt differs (-built +decoded):\n%s", diff)
			}
			if back.IsZero() {
				t.Errorf("a set prompt reports IsZero")
			}
		})
	}
}

func TestPromptObjectRecordings(t *testing.T) {
	tests := map[string]struct{ fixture string }{
		"success: mixed text and token": {"object_mixed"},
		"success: token IDs":            {"object_tokens"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/completion/" + tt.fixture
			completionCheckWire[CompletionRequest](t, jsontest.ReadFile(t, prefix+".req.json"))
			completionCheckWire[CompletionResponse](t, jsontest.ReadFile(t, prefix+".resp.json"))
		})
	}
}

func TestPromptAccessors(t *testing.T) {
	p := MixedPrompt(TextPart("a"), TokenPart(7))
	if s, ok := p.Parts()[0].Text(); !ok || s != "a" {
		t.Errorf("Parts()[0].Text() = %q, %t", s, ok)
	}
	if tok, ok := p.Parts()[1].Token(); !ok || tok != 7 {
		t.Errorf("Parts()[1].Token() = %d, %t", tok, ok)
	}
	if _, ok := p.Parts()[0].Token(); ok {
		t.Errorf("a text part reports a token")
	}
	text, ok := MultimodalPrompt("o").Object().PromptString.Text()
	if TextPrompt("x").Text() != "x" || len(TokenPrompt(1, 2).Tokens()) != 2 || !ok || text != "o" || len(BatchPrompt(TextPrompt("a")).Batch()) != 1 {
		t.Errorf("accessors do not return the built content")
	}
	if !(Prompt{}).IsZero() || !(Prompt{}).Equal(Prompt{}) || TextPrompt("a").Equal(TokenPrompt(1)) {
		t.Errorf("IsZero or Equal is wrong for the zero prompt")
	}
}

func TestPromptJSONErrors(t *testing.T) {
	tests := map[string]struct {
		prompt Prompt
		wire   string
	}{
		"error: parts with media":          {prompt: ObjectPrompt(PromptObject{PromptString: TokenizeParts(TokenPart(1)), MultimodalData: [][]byte{}})},
		"error: mixed without a token":     {prompt: MixedPrompt(TextPart("a"), TextPart("b"))},
		"error: nested batch":              {prompt: BatchPrompt(BatchPrompt(TextPrompt("a")))},
		"error: zero in batch":             {prompt: BatchPrompt(Prompt{})},
		"error: parts with empty media":    {wire: `{"prompt_string":[1],"multimodal_data":[]}`},
		"error: parts with null media":     {wire: `{"prompt_string":[1],"multimodal_data":null}`},
		"error: nested object parts":       {wire: `{"prompt_string":[[1]]}`},
		"error: number":                    {wire: `12`},
		"error: bool":                      {wire: `true`},
		"error: object inside mixed":       {wire: `[1,{"a":1}]`},
		"error: float token":               {wire: `[1.5,2]`},
		"error: batch element of strings":  {wire: `[["a","b"]]`},
		"error: batch element of a number": {wire: `["a",true]`},
		"error: null batch element":        {wire: `["x",null]`},
		"error: only a null element":       {wire: `[null]`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if tt.wire == "" {
				if out, err := json.Marshal(tt.prompt); err == nil {
					t.Errorf("Marshal succeeded with %s", out)
				}
				return
			}
			var p Prompt
			if err := json.Unmarshal([]byte(tt.wire), &p); err == nil {
				t.Errorf("Unmarshal(%s) succeeded with kind %d", tt.wire, p.Kind())
			}
		})
	}
}

func TestLogitBiasJSON(t *testing.T) {
	tests := map[string]struct {
		bias    LogitBias
		json    string
		wantErr bool
	}{
		"success: array form with ban and text": {
			bias: LogitBiasArray(
				LogitBiasEntry{Target: TokenTarget(15043), Bias: 1.5},
				LogitBiasEntry{Target: TextTarget("Hello, World!"), Bias: -0.5},
				LogitBiasEntry{Target: TokenTarget(2), Ban: true},
			),
			json: `[[15043,1.5],["Hello, World!",-0.5],[2,false]]`,
		},
		"success: object form keeps order": {
			bias: LogitBiasObject(
				LogitBiasEntry{Target: TokenTarget(30), Bias: 2},
				LogitBiasEntry{Target: TextTarget("Hello"), Ban: true},
				LogitBiasEntry{Target: TokenTarget(1), Bias: -100},
			),
			json: `{"30":2,"Hello":false,"1":-100}`,
		},
		"success: empty string target stays a string": {
			bias: LogitBiasArray(LogitBiasEntry{Target: TextTarget(""), Bias: 1}),
			json: `[["",1]]`,
		},
		"success: token 0 stays a token": {
			bias: LogitBiasArray(LogitBiasEntry{Target: TokenTarget(0), Bias: 1}),
			json: `[[0,1]]`,
		},
		"success: object keys keep their spelling": {
			bias: LogitBiasObject(
				LogitBiasEntry{Target: TextTarget("01"), Bias: 1},
				LogitBiasEntry{Target: TokenTarget(1), Bias: 2},
				LogitBiasEntry{Target: TextTarget("+1"), Bias: 3},
				LogitBiasEntry{Target: TokenTarget(-1), Bias: 4},
			),
			json: `{"01":1,"1":2,"+1":3,"-1":4}`,
		},
		"success: empty object":     {bias: LogitBiasObject(), json: `{}`},
		"success: empty array":      {bias: LogitBiasArray(), json: `[]`},
		"error: duplicate member":   {json: `{"1":1,"1":2}`, wantErr: true},
		"error: true is not a bias": {json: `{"1":true}`, wantErr: true},
		"error: pair of three":      {json: `[[1,2,3]]`, wantErr: true},
		"error: target object":      {json: `[[{},1]]`, wantErr: true},
		"error: scalar":             {json: `5`, wantErr: true},
		"error: string bias":        {json: `[[1,"x"]]`, wantErr: true},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if tt.wantErr {
				var b LogitBias
				if err := json.Unmarshal([]byte(tt.json), &b); err == nil {
					t.Errorf("Unmarshal(%s) succeeded", tt.json)
				}
				return
			}
			got, err := json.Marshal(tt.bias)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.json {
				t.Errorf("Marshal = %s, want %s", got, tt.json)
			}
			var back LogitBias
			if err := json.Unmarshal([]byte(tt.json), &back); err != nil {
				t.Fatalf("Unmarshal: %v", err)
			}
			if diff := gocmp.Diff(tt.bias, back); diff != "" {
				t.Errorf("decoded bias differs (-built +decoded):\n%s", diff)
			}
			if back.IsZero() || back.IsObject() != tt.bias.IsObject() || len(back.Entries()) != len(tt.bias.Entries()) {
				t.Errorf("accessors disagree: zero=%t object=%t entries=%d", back.IsZero(), back.IsObject(), len(back.Entries()))
			}
		})
	}
	if out, err := json.Marshal(LogitBiasObject(
		LogitBiasEntry{Target: TokenTarget(1), Bias: 1},
		LogitBiasEntry{Target: TextTarget("1"), Bias: 2},
	)); err == nil {
		t.Errorf("Marshal of two entries spelling the member \"1\" succeeded: %s", out)
	}
	tok, isTok := TokenTarget(7).Token()
	text, isText := TextTarget("x").Text()
	if tok != 7 || !isTok || text != "x" || !isText {
		t.Errorf("target accessors: %d %t %q %t", tok, isTok, text, isText)
	}
	if _, ok := TokenTarget(7).Text(); ok {
		t.Errorf("a token target reports a text")
	}
}

func TestStringOrSliceJSON(t *testing.T) {
	tests := map[string]struct {
		v    StringOrSlice
		json string
	}{
		"success: single":         {v: StringValue("\n"), json: `"\n"`},
		"success: slice":          {v: StringSlice("a", "b"), json: `["a","b"]`},
		"success: one-elem slice": {v: StringSlice("a"), json: `["a"]`},
		"success: empty slice":    {v: StringSlice(), json: `[]`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := json.Marshal(tt.v)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.json {
				t.Errorf("Marshal = %s, want %s", got, tt.json)
			}
			var back StringOrSlice
			if err := json.Unmarshal(got, &back); err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tt.v, back); diff != "" {
				t.Errorf("round trip (-want +got):\n%s", diff)
			}
			if back.IsZero() || back.IsSingle() != tt.v.IsSingle() || len(back.Strings()) != len(tt.v.Strings()) {
				t.Errorf("accessors disagree")
			}
		})
	}
	var v StringOrSlice
	if err := json.Unmarshal([]byte(`3`), &v); err == nil {
		t.Errorf("Unmarshal accepted a number")
	}
	if err := json.Unmarshal([]byte(`null`), &v); err != nil || !v.IsZero() {
		t.Errorf("null must decode to the zero value: %v", err)
	}
}

// samplingRequest embeds SamplingParams the way endpoint request types do.
type samplingRequest struct {
	Prompt Prompt `json:"prompt,omitzero"`
	SamplingParams
	Stream *bool                     `json:"stream,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

func TestSamplingParamsEmbedding(t *testing.T) {
	req := samplingRequest{
		Prompt:      TextPrompt("Building a website can be done in"),
		Temperature: new(0.0),
		NPredict:    new(0),
		CachePrompt: new(false),
		Stop:        StringSlice("\n"),
		LogitBias:   LogitBiasObject(LogitBiasEntry{Target: TokenTarget(1), Bias: -1}),
		Samplers:    StringValue("kpt"),
		LoRA:        []LoRAWeight{{ID: 0, Scale: 0.5}},
		JSONSchema:  jsontext.Value(`{"type":"object"}`),
		Extra:       map[string]jsontext.Value{"future_param": jsontext.Value(`[1]`)},
	}
	got, err := json.Marshal(req, jsonOptions)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"prompt":"Building a website can be done in","n_predict":0,"cache_prompt":false,"temperature":0,` +
		`"samplers":"kpt","logit_bias":{"1":-1},"stop":["\n"],"json_schema":{"type":"object"},` +
		`"lora":[{"id":0,"scale":0.5}],"future_param":[1]}`
	if string(got) != want {
		t.Errorf("Marshal =\n%s\nwant\n%s", got, want)
	}
	jsontest.RoundTrip[samplingRequest](t, got)
	jsontest.InjectUnknown[samplingRequest](t, got, jsontest.SkipObjects(
		"/logit_bias",  // a map from token to bias, not a record
		"/json_schema", // raw JSON kept as is, so an injected member survives anyway
	))
	jsontest.CheckFallbacks[samplingRequest](t)
}

func TestSharedResponseTypes(t *testing.T) {
	// Member sets follow server_slot_stats::to_json, usage_json_oaicompat, result_prompt_progress::to_json and
	// probs_vector_to_json in llama.cpp v1.1.0.
	tests := map[string]struct {
		check func(t *testing.T, data []byte)
		data  string
	}{
		"Timings": {
			check: func(t *testing.T, d []byte) {
				jsontest.RoundTrip[Timings](t, d)
				jsontest.NoUnknown[Timings](t, d, nil)
			},
			data: `{"cache_n":0,"prompt_n":5,"prompt_ms":1.5,"prompt_per_token_ms":0.3,"prompt_per_second":3333.3,"predicted_n":8,"predicted_ms":2.5,"predicted_per_token_ms":0.31,"predicted_per_second":3200,"draft_n":4,"draft_n_accepted":3}`,
		},
		"Usage": {
			check: func(t *testing.T, d []byte) { jsontest.RoundTrip[Usage](t, d); jsontest.NoUnknown[Usage](t, d, nil) },
			data:  `{"completion_tokens":8,"prompt_tokens":5,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":0}}`,
		},
		"PromptProgress": {
			check: func(t *testing.T, d []byte) {
				jsontest.RoundTrip[PromptProgress](t, d)
				jsontest.NoUnknown[PromptProgress](t, d, nil)
			},
			data: `{"total":10,"cache":2,"processed":5,"time_ms":1.25}`,
		},
		"TokenProb": {
			check: func(t *testing.T, d []byte) {
				jsontest.RoundTrip[[]TokenProb](t, d)
				jsontest.InjectUnknown[[]TokenProb](t, d)
				jsontest.NoUnknown[[]TokenProb](t, d, nil)
			},
			data: `[{"id":5,"token":" a","bytes":[32,97],"logprob":-0.1,"top_logprobs":[{"id":5,"token":" a","bytes":[32,97],"logprob":-0.1}]},` +
				`{"id":6,"token":"b","bytes":[98],"prob":0.9,"top_probs":[{"id":6,"token":"b","bytes":[98],"prob":0.9}]}]`,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) { tt.check(t, []byte(tt.data)) })
	}
	jsontest.CheckFallbacks[Timings](t)
	jsontest.CheckFallbacks[Usage](t)
	jsontest.CheckFallbacks[PromptProgress](t)
	jsontest.CheckFallbacks[TokenProb](t)
	jsontest.CheckFallbacks[LoRAWeight](t)
	jsontest.CheckFallbacks[HealthResponse](t)
}

func FuzzPrompt(f *testing.F) {
	for _, fixture := range []string{"object_mixed", "object_tokens"} {
		var req struct {
			Prompt jsontext.Value `json:"prompt"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/completion/"+fixture+".req.json"), &req); err != nil {
			f.Fatal(err)
		}
		f.Add(string(req.Prompt))
	}
	for _, seed := range []string{
		`"text"`, `[1,2,3]`, `[1,"a",2]`, `{"prompt_string":"x","multimodal_data":["AA=="]}`,
		`["a",[1,2],{"prompt_string":"b"}]`, `[]`, `null`, `[[1,"a"]]`, `["x",null]`, `[null]`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, in string) {
		var p Prompt
		if err := json.Unmarshal([]byte(in), &p); err != nil {
			return
		}
		out, err := json.Marshal(p, jsonOptions)
		if err != nil {
			t.Fatalf("decoded %s but cannot encode it: %v", in, err)
		}
		var back Prompt
		if err := json.Unmarshal(out, &back); err != nil {
			t.Fatalf("re-decoding %s: %v", out, err)
		}
		if !p.Equal(back) {
			t.Fatalf("round trip changed the prompt: %s -> %s", in, out)
		}
	})
}

func FuzzLogitBias(f *testing.F) {
	for _, seed := range []string{
		`[[1,2.5],["x",false]]`, `{"1":-1,"y":false}`, `[]`, `{}`, `[["",1]]`, `{"01":1,"1":2}`, `{"-0":1,"+5":2}`,
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, in string) {
		var b LogitBias
		if err := json.Unmarshal([]byte(in), &b); err != nil {
			return
		}
		out, err := json.Marshal(b)
		if err != nil {
			t.Fatalf("decoded %s but cannot encode it: %v", in, err)
		}
		var back LogitBias
		if err := json.Unmarshal(out, &back); err != nil {
			t.Fatalf("re-decoding %s: %v", out, err)
		}
		if !b.Equal(back) {
			t.Fatalf("round trip changed the bias: %s -> %s", in, out)
		}
		// The encoding must say what the input said: a string target stays a string, a key keeps its spelling.
		jsontest.RoundTrip[LogitBias](t, []byte(in))
		again, err := json.Marshal(back)
		if err != nil || !bytes.Equal(again, out) {
			t.Fatalf("re-encoding changed the bias: %s -> %s -> %s", in, out, again)
		}
	})
}

// The server accepts "samplers" as an array of names or as a string of sampler characters
// (tools/server/server-schema.cpp of llama.cpp v1.1.0).
func TestSamplersForms(t *testing.T) {
	tests := map[string]struct {
		json string
		want StringOrSlice
	}{
		"success: compact string": {json: `{"samplers":"kpt"}`, want: StringValue("kpt")},
		"success: names":          {json: `{"samplers":["top_k","top_p"]}`, want: StringSlice("top_k", "top_p")},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got := jsontest.RoundTrip[samplingRequest](t, []byte(tt.json))
			if diff := gocmp.Diff(tt.want, got.Samplers); diff != "" {
				t.Errorf("Samplers (-want +got):\n%s", diff)
			}
		})
	}
}

// The speculative.* request fields are compiled out of the server's request schema at v1.1.0, so the request type
// does not offer them; a member sent anyway is kept as an unknown member.
func TestSamplingParamsHasNoIgnoredSpeculativeFields(t *testing.T) {
	for f := range reflect.TypeFor[SamplingParams]().Fields() {
		if strings.HasPrefix(f.Tag.Get("json"), "speculative.") {
			t.Errorf("SamplingParams.%s maps %q, which llama-server v1.1.0 ignores", f.Name, f.Tag.Get("json"))
		}
	}
	got := jsontest.RoundTrip[samplingRequest](t, []byte(`{"speculative.n_max":8}`))
	if _, ok := got.Extra["speculative.n_max"]; !ok {
		t.Errorf("speculative.n_max was not kept as an unknown member: %+v", got)
	}
}

// TestNoUnknownSeesInsidePrompt checks that the unknown-member gate reaches the object form of a Prompt, which is
// kept in an unexported field.
func TestNoUnknownSeesInsidePrompt(t *testing.T) {
	tests := map[string]struct {
		data string
		want string
	}{
		"error: object prompt":         {data: `{"prompt_string":"hello","future":1}`, want: "/: future"},
		"error: object inside a batch": {data: `["a",{"prompt_string":"b","later":2}]`, want: "/1: later"},
		"success: no unknown member":   {data: `{"prompt_string":"hello"}`},
		"success: text has no members": {data: `"hello"`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			msgs, _ := jsontest.Failures(t, func(tb testing.TB) { jsontest.NoUnknown[Prompt](tb, []byte(tt.data), nil) })
			if tt.want == "" {
				if len(msgs) != 0 {
					t.Errorf("unexpected failures: %v", msgs)
				}
				return
			}
			if len(msgs) != 1 || !strings.Contains(msgs[0], tt.want) {
				t.Errorf("NoUnknown[Prompt](%s) failures = %v, want one naming %q", tt.data, msgs, tt.want)
			}
		})
	}
}
