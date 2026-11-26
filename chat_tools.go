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
	// Function is the required declaration of the callable name and argument schema.
	Function *FunctionDefinition `json:"function,omitzero"`
	// DeferLoading requests deferred loading; defaults to false and can be overridden by Function.DeferLoading.
	DeferLoading *bool `json:"defer_loading,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionDefinition describes a function tool.
type FunctionDefinition struct {
	// Name is the required callable name used to match tool choices and model-generated calls.
	Name string `json:"name"`
	// Description tells the model what the function does; the server uses an empty string if omitted.
	Description *string `json:"description,omitzero"`
	// Parameters is the argument schema, defaulting to {}; SchemaFor builds one from a Go type.
	Parameters jsontext.Value `json:"parameters,omitzero"`
	// Strict asks for arguments that match the schema exactly; llama.cpp v1.1.0 accepts but ignores it.
	Strict *bool `json:"strict,omitzero"`
	// DeferLoading overrides the tool-level setting when present, including when false.
	DeferLoading *bool `json:"defer_loading,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ToolChoice is the "tool_choice" of a request: a mode, a named function, or an allowlist of functions.
// Build one with ToolChoiceMode, ToolChoiceFunction, or ToolChoiceAllowedTools. An object decoded with a
// type other than "function" or "allowed_tools" is kept as received and re-encoded verbatim.
// Encoding the value after modifying a slice returned by its accessors is unspecified.
type ToolChoice struct {
	mode     string
	function *NamedToolChoice
	allowed  *AllowedToolChoice
	unknown  *jsontext.Value
}

// NamedToolChoice is the object form of a ToolChoice that names one function.
type NamedToolChoice struct {
	// Type is "function".
	Type string `json:"type"`
	// Function names the function to call.
	Function FunctionName `json:"function"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// FunctionName names the function of a NamedToolChoice.
type FunctionName struct {
	// Name must match exactly one declared tool when used to force a function call.
	Name string `json:"name"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AllowedToolChoice is the object form of a ToolChoice that limits the model to a set of declared tools.
type AllowedToolChoice struct {
	// Type is "allowed_tools".
	Type string `json:"type"`
	// AllowedTools holds the selection mode and function references.
	AllowedTools *AllowedTools `json:"allowed_tools,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AllowedTools holds the selection mode and function references of an AllowedToolChoice.
type AllowedTools struct {
	// Mode is "auto" or "required".
	Mode string `json:"mode"`
	// Tools names the allowed functions. The server requires a nonempty list in "required" mode.
	Tools []NamedToolChoice `json:"tools,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
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

// Unknown returns a read-only view of the raw JSON of an unrecognized object, or nil.
func (c ToolChoice) Unknown() jsontext.Value {
	if c.unknown == nil {
		return nil
	}
	return *c.unknown
}

// IsZero reports whether the tool choice is unset.
func (c ToolChoice) IsZero() bool { return c == (ToolChoice{}) }

// Equal reports whether two tool choices have the same form and content.
func (c ToolChoice) Equal(o ToolChoice) bool {
	return reflect.DeepEqual(c, o)
}

// MarshalJSONTo implements json.MarshalerTo.
func (c ToolChoice) MarshalJSONTo(enc *jsontext.Encoder) error {
	if c.unknown != nil {
		return enc.WriteValue(*c.unknown)
	}
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
		switch tag.Type {
		case "allowed_tools":
			a := new(AllowedToolChoice)
			if err := json.Unmarshal(v, a, opts); err != nil {
				return err
			}
			*c = ToolChoice{allowed: a}
		case "function":
			n := new(NamedToolChoice)
			if err := json.Unmarshal(v, n, opts); err != nil {
				return err
			}
			*c = ToolChoice{function: n}
		default:
			*c = ToolChoice{unknown: new(v.Clone())}
		}
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

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// JSONSchemaFormat is the schema of a "json_schema" ResponseFormat.
type JSONSchemaFormat struct {
	// Name labels the schema for OpenAI compatibility; llama.cpp does not use it to build the grammar.
	Name string `json:"name,omitzero"`
	// Description is optional schema metadata; llama.cpp does not pass it to the grammar converter.
	Description *string `json:"description,omitzero"`
	// Schema becomes a GBNF grammar; omission permits any JSON object, and SchemaFor builds one from a Go type.
	Schema jsontext.Value `json:"schema,omitzero"`
	// Strict is an OpenAI compatibility flag; llama.cpp applies its schema-to-grammar conversion regardless.
	Strict *bool `json:"strict,omitzero"`

	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}
