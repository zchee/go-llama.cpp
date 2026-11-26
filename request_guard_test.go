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
	"net/http"
	"strings"
	"testing"
)

func TestRequestExtra(t *testing.T) {
	type record struct {
		Flag  *bool                     `json:"enabled,omitzero"`
		Extra map[string]jsontext.Value `json:",embed"`
	}
	type nested struct {
		Child *record
		Items []record
		Table map[string]record
	}
	extra := map[string]jsontext.Value{"enabled": jsontext.Value(`false`)}
	child := record{Extra: extra}
	type Embedded struct{ *record }
	type quoted struct {
		Option *bool                     `json:"'a,b',omitzero"`
		Extra  map[string]jsontext.Value `json:",embed"`
	}
	type shadow struct {
		record
		Preferred *bool `json:"enabled,omitzero"`
	}
	type rawFallback struct {
		record
		Raw jsontext.Value `json:",embed"`
	}
	type childFallback struct {
		Extra map[string]record `json:",embed"`
	}
	tests := map[string]struct {
		body            any
		typ, key, field string
	}{
		"success: raw fallback shadows embedded fallback": {body: rawFallback{record: child}},
		"error: fallback value is a typed child":          {childFallback{map[string]record{"child": child}}, "record", "enabled", "Flag"},
		"error: quoted tag with comma":                    {quoted{Extra: map[string]jsontext.Value{"a,b": jsontext.Value(`false`)}}, "quoted", "a,b", "Option"},
		"error: shallow member takes precedence":          {shadow{record: child}, "shadow", "enabled", "Preferred"},
		"error: fallback in embedded pointer":             {Embedded{&child}, "Embedded", "enabled", "Flag"},
		"success: nil embedded pointer":                   {body: Embedded{}},
		"success: unexported state is not sent": {body: struct {
			Name  string
			state record
		}{state: child}},
		"error: caller-defined type":                 {child, "record", "enabled", "Flag"},
		"error: promoted member":                     {&CompletionRequest{Extra: map[string]jsontext.Value{"n_predict": jsontext.Value(`1`)}}, "CompletionRequest", "n_predict", "NPredict"},
		"error: nested object":                       {nested{Child: &child}, "record", "enabled", "Flag"},
		"error: slice element":                       {nested{Items: []record{{}, child}}, "record", "enabled", "Flag"},
		"error: map element":                         {nested{Table: map[string]record{"entry": child}}, "record", "enabled", "Flag"},
		"error: object behind custom prompt encoder": {CompletionRequest{Prompt: ObjectPrompt(PromptObject{PromptString: TokenizeText("x"), Extra: map[string]jsontext.Value{"multimodal_data": jsontext.Value(`[]`)}})}, "PromptObject", "multimodal_data", "MultimodalData"},
		"success: exact names only":                  {body: record{Extra: map[string]jsontext.Value{"Enabled": jsontext.Value(`1`)}}},
		"success: top-level map is unguarded":        {body: map[string]record{"entry": child}},
	}
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/completion", tt.body)
			if tt.field == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if req != nil || err == nil {
				t.Fatalf("NewRequest returned request=%t, error=%v", req != nil, err)
			}
			for _, want := range []string{tt.typ, tt.key, tt.field} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not identify %q", err, want)
				}
			}
		})
	}
}

func TestRequestExtraCycle(t *testing.T) {
	type node struct {
		Next  *node
		Value any
	}
	n := new(node)
	n.Next = n
	m := map[string]any{}
	m["cycle"] = m
	s := make([]any, 1)
	s[0] = s
	tests := map[string]struct{ body *node }{
		"error: pointer cycle": {n},
		"error: map cycle":     {&node{Value: m}},
		"error: slice cycle":   {&node{Value: s}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if _, _, err := encodeBody(tt.body); err == nil || !strings.Contains(err.Error(), "cycle") {
				t.Errorf("encodeBody error = %v, want a JSON cycle error", err)
			}
		})
	}
}

func BenchmarkRequestWithoutExtra(b *testing.B) {
	body := &CompletionRequest{Prompt: TextPrompt("Once"), NPredict: new(1)}
	b.Run("checkRequestExtra", func(b *testing.B) {
		if err := checkRequestExtra(body); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		for b.Loop() {
			if err := checkRequestExtra(body); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("marshal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := json.Marshal(body, jsonOptions); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("encodeBody", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, _, err := encodeBody(body); err != nil {
				b.Fatal(err)
			}
		}
	})
}
