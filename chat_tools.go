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
	"fmt"
	"reflect"
)

// Tool is a tool the model may call. llama.cpp needs --jinja for tools.
type Tool struct {
	// Type is "function".
	Type string `json:"type"`
	// Function describes the function.
	Function *FunctionDefinition `json:"function,omitzero"`
	// DeferLoading marks the tool for deferred loading.
	DeferLoading *bool `json:"defer_loading,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionDefinition describes a function tool.
type FunctionDefinition struct {
	// Name is the function name.
	Name string `json:"name"`
	// Description tells the model what the function does.
	Description *string `json:"description,omitzero"`
	// Parameters is the JSON schema of the arguments; SchemaFor builds one from a Go type.
	Parameters jsontext.Value `json:"parameters,omitzero"`
	// Strict asks for arguments that match the schema exactly; llama.cpp v1.1.0 accepts but ignores it.
	Strict *bool `json:"strict,omitzero"`
	// DeferLoading overrides the tool-level setting when present, including when false.
	DeferLoading *bool `json:"defer_loading,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolChoice is the "tool_choice" of a request: a mode, a named function, or an allowlist of functions.
// Build one with ToolChoiceMode, ToolChoiceFunction, or ToolChoiceAllowedTools.
type ToolChoice struct {
	mode     string
	function *NamedToolChoice
	allowed  *AllowedToolChoice
}

// NamedToolChoice is the object form of a ToolChoice that names one function.
type NamedToolChoice struct {
	// Type is "function".
	Type string `json:"type"`
	// Function names the function to call.
	Function FunctionName `json:"function"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionName names the function of a NamedToolChoice.
type FunctionName struct {
	// Name is the function name.
	Name string `json:"name"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// AllowedToolChoice is the object form of a ToolChoice that limits the model to a set of declared tools.
type AllowedToolChoice struct {
	// Type is "allowed_tools".
	Type string `json:"type"`
	// AllowedTools holds the selection mode and function references.
	AllowedTools *AllowedTools `json:"allowed_tools,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// AllowedTools holds the selection mode and function references of an AllowedToolChoice.
type AllowedTools struct {
	// Mode is "auto" or "required".
	Mode string `json:"mode"`
	// Tools names the allowed functions. The server requires a nonempty list in "required" mode.
	Tools []NamedToolChoice `json:"tools,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolChoiceMode returns a tool choice given as a mode: "auto", "none" or "required".
func ToolChoiceMode(mode string) ToolChoice { return ToolChoice{mode: mode} }

// ToolChoiceFunction returns a tool choice that forces a call of the named function.
func ToolChoiceFunction(name string) ToolChoice {
	return ToolChoice{function: &NamedToolChoice{Type: "function", Function: FunctionName{Name: name}}}
}

// ToolChoiceAllowedTools returns a tool choice that limits the model to the named functions in the given mode.
func ToolChoiceAllowedTools(mode string, names ...string) ToolChoice {
	tools := make([]NamedToolChoice, len(names))
	for i, name := range names {
		tools[i] = NamedToolChoice{Type: "function", Function: FunctionName{Name: name}}
	}
	return ToolChoice{allowed: &AllowedToolChoice{Type: "allowed_tools", AllowedTools: &AllowedTools{Mode: mode, Tools: tools}}}
}

// Mode returns the mode of a tool choice given as a string.
func (c ToolChoice) Mode() string { return c.mode }

// Function returns the object of a tool choice given as a named function.
func (c ToolChoice) Function() *NamedToolChoice { return c.function }

// AllowedTools returns the object of a tool choice given as an allowlist.
func (c ToolChoice) AllowedTools() *AllowedToolChoice { return c.allowed }

// IsZero reports whether the tool choice is unset.
func (c ToolChoice) IsZero() bool { return c.mode == "" && c.function == nil && c.allowed == nil }

// Equal reports whether two tool choices have the same form and content.
func (c ToolChoice) Equal(o ToolChoice) bool {
	return c.mode == o.mode && reflect.DeepEqual(c.function, o.function) && reflect.DeepEqual(c.allowed, o.allowed)
}

// MarshalJSONTo implements json.MarshalerTo.
func (c ToolChoice) MarshalJSONTo(enc *jsontext.Encoder) error {
	if c.allowed != nil {
		return json.MarshalEncode(enc, c.allowed)
	}
	if c.function != nil {
		return json.MarshalEncode(enc, c.function)
	}
	return enc.WriteToken(jsontext.String(c.mode))
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom.
func (c *ToolChoice) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch kind := dec.PeekKind(); kind {
	case '"':
		var s string
		if err := json.UnmarshalDecode(dec, &s); err != nil {
			return err
		}
		*c = ToolChoiceMode(s)
	case '{':
		v, err := dec.ReadValue()
		if err != nil {
			return err
		}
		var tag struct {
			Type string `json:"type"`
		}
		opts := dec.Options()
		if err := json.Unmarshal(v, &tag, opts, json.RejectUnknownMembers(false)); err != nil {
			return err
		}
		if tag.Type == "allowed_tools" {
			a := new(AllowedToolChoice)
			if err := json.Unmarshal(v, a, opts); err != nil {
				return err
			}
			*c = ToolChoice{allowed: a}
			return nil
		}
		// Preserve the legacy handling of object types other than "allowed_tools".
		n := new(NamedToolChoice)
		if err := json.Unmarshal(v, n, opts); err != nil {
			return err
		}
		*c = ToolChoice{function: n}
	default:
		return fmt.Errorf("openai: tool_choice must be a string or an object, got %s", kind)
	}
	return nil
}

// ResponseFormat is the "response_format" of a request. llama.cpp turns a schema into a grammar; it reads
// JSONSchema.Schema for "json_schema" and Schema for "json_object" or a "json_schema" without the wrapper.
type ResponseFormat struct {
	// Type is "text", "json_object" or "json_schema".
	Type string `json:"type"`
	// JSONSchema is the schema of a "json_schema" format.
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitzero"`
	// Schema is a schema given directly, a llama.cpp extension.
	Schema jsontext.Value `json:"schema,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}

// JSONSchemaFormat is the schema of a "json_schema" ResponseFormat.
type JSONSchemaFormat struct {
	// Name names the schema.
	Name string `json:"name,omitzero"`
	// Description describes the schema.
	Description *string `json:"description,omitzero"`
	// Schema is the JSON schema; SchemaFor builds one from a Go type.
	Schema jsontext.Value `json:"schema,omitzero"`
	// Strict asks for output that matches the schema exactly.
	Strict *bool `json:"strict,omitzero"`

	Extra map[string]jsontext.Value `json:",embed"`
}
