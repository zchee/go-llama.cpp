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
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
)

type weather struct {
	City  string   `json:"city" jsonschema:"the city name"`
	Days  int      `json:"days,omitzero"`
	Units []string `json:"units"`
}

type withFallback struct {
	Name  string                    `json:"name"`
	Extra map[string]jsontext.Value `json:",embed"`
}

type flattened struct {
	Parts struct {
		X int `json:"x"`
	} `json:",embed"`
}

type deepFallback struct {
	Items []*struct {
		Inner map[string]withFallback `json:"inner"`
	} `json:"items"`
}

type selfRef struct {
	Name     string     `json:"name"`
	Children []*selfRef `json:"children,omitzero"`
}

func TestSchemaFor(t *testing.T) {
	got, err := SchemaFor[weather]()
	if err != nil {
		t.Fatalf("SchemaFor[weather]: %v", err)
	}
	var schema map[string]any
	if err := json.Unmarshal(got, &schema); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"city", "units"},
		"properties": map[string]any{
			"city":  map[string]any{"type": "string", "description": "the city name"},
			"days":  map[string]any{"type": "integer"},
			"units": map[string]any{"type": []any{"null", "array"}, "items": map[string]any{"type": "string"}},
		},
	}
	if diff := gocmp.Diff(want, schema); diff != "" {
		t.Errorf("SchemaFor[weather] (-want +got):\n%s\nschema: %s", diff, got)
	}
}

func TestSchemaForRejectsEmbed(t *testing.T) {
	tests := map[string]struct {
		schema   func() (jsontext.Value, error)
		wantPath string
	}{
		"error: fallback map":                  {schema: SchemaFor[withFallback], wantPath: "llama.withFallback.Extra"},
		"error: flattened struct":              {schema: SchemaFor[flattened], wantPath: "llama.flattened.Parts"},
		"error: fallback deep in slices/maps":  {schema: SchemaFor[deepFallback], wantPath: "llama.withFallback.Extra"},
		"error: library request type":          {schema: SchemaFor[HealthResponse], wantPath: "llama.HealthResponse.Extra"},
		"error: pointer to a type with embeds": {schema: SchemaFor[*withFallback], wantPath: "llama.withFallback.Extra"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := tt.schema()
			if err == nil {
				t.Fatalf("SchemaFor accepted a type using embed: %s", got)
			}
			if !strings.Contains(err.Error(), tt.wantPath) || !strings.Contains(err.Error(), "embed") {
				t.Errorf("error = %v, want it to name %s and the embed option", err, tt.wantPath)
			}
		})
	}
}

func TestSchemaForGeneratorErrors(t *testing.T) {
	if _, err := SchemaFor[selfRef](); err == nil {
		t.Errorf("SchemaFor accepted a recursive type, which the generator reports as a cycle")
	}
	if _, err := SchemaFor[struct{ F func() }](); err == nil {
		t.Errorf("SchemaFor accepted a func field")
	}
}

func TestHasJSONOption(t *testing.T) {
	tests := map[string]struct {
		tag  string
		want bool
	}{
		"success: bare embed":    {tag: ",embed", want: true},
		"success: after another": {tag: "x,omitzero,embed", want: true},
		"no: name only":          {tag: "embed", want: false},
		"no: other options":      {tag: "x,omitzero", want: false},
		"no: empty":              {tag: "", want: false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := hasJSONOption(tt.tag, "embed"); got != tt.want {
				t.Errorf("hasJSONOption(%q) = %t, want %t", tt.tag, got, tt.want)
			}
		})
	}
}
