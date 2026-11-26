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

package realtime

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
)

// EventType identifies a realtime protocol event.
type EventType string

// Client event types accepted by llama-server.
const (
	SessionUpdate              EventType = "session.update"
	TranscriptionSessionUpdate EventType = "transcription_session.update"
	InputAudioBufferAppend     EventType = "input_audio_buffer.append"
	InputAudioBufferCommit     EventType = "input_audio_buffer.commit"
	InputAudioBufferClear      EventType = "input_audio_buffer.clear"
	ConversationItemCreate     EventType = "conversation.item.create"
	ResponseCreate             EventType = "response.create"
	ResponseCancel             EventType = "response.cancel"
)

// Server event types emitted for transcription, conversation and full duplex sessions.
const (
	SessionCreated                     EventType = "session.created"
	SessionUpdated                     EventType = "session.updated"
	InputAudioBufferCleared            EventType = "input_audio_buffer.cleared"
	InputAudioBufferCommitted          EventType = "input_audio_buffer.committed"
	ConversationItemAdded              EventType = "conversation.item.added"
	ConversationItemDone               EventType = "conversation.item.done"
	TranscriptionDelta                 EventType = "conversation.item.input_audio_transcription.delta"
	TranscriptionCompleted             EventType = "conversation.item.input_audio_transcription.completed"
	ResponseCreated                    EventType = "response.created"
	ResponseOutputItemAdded            EventType = "response.output_item.added"
	ResponseOutputItemDone             EventType = "response.output_item.done"
	ResponseContentPartAdded           EventType = "response.content_part.added"
	ResponseContentPartDone            EventType = "response.content_part.done"
	ResponseOutputTextDelta            EventType = "response.output_text.delta"
	ResponseOutputTextDone             EventType = "response.output_text.done"
	ResponseOutputAudioDelta           EventType = "response.output_audio.delta"
	ResponseOutputAudioDone            EventType = "response.output_audio.done"
	ResponseFunctionCallArgumentsDelta EventType = "response.function_call_arguments.delta"
	ResponseFunctionCallArgumentsDone  EventType = "response.function_call_arguments.done"
	ResponseDone                       EventType = "response.done"
	ErrorEvent                         EventType = "error"
)

// Nullable distinguishes a missing member (zero value), explicit JSON null and a value.
type Nullable[T any] struct {
	Value *T
	Null  bool
}

// MarshalJSONTo writes the value or JSON null.
func (n Nullable[T]) MarshalJSONTo(enc *jsontext.Encoder) error {
	if n.Null && n.Value != nil {
		return errors.New("realtime: nullable member has both null and a value")
	}
	return json.MarshalEncode(enc, n.Value)
}

// UnmarshalJSONFrom records whether the JSON value was explicitly null.
func (n *Nullable[T]) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var v *T
	if err := json.UnmarshalDecode(dec, &v); err != nil {
		return err
	}
	*n = Nullable[T]{Value: v, Null: v == nil}
	return nil
}

// MaxOutputTokens limits a session or response to a positive number of tokens or UnlimitedTokens.
type MaxOutputTokens int64

// UnlimitedTokens is the wire value "inf".
const UnlimitedTokens MaxOutputTokens = -1

// MarshalJSONTo writes a positive integer or "inf".
func (m MaxOutputTokens) MarshalJSONTo(enc *jsontext.Encoder) error {
	if m == UnlimitedTokens {
		return enc.WriteToken(jsontext.String("inf"))
	}
	if m < 1 {
		return errors.New("realtime: max_output_tokens must be positive or inf")
	}
	return enc.WriteToken(jsontext.Int(int64(m)))
}

// UnmarshalJSONFrom reads a positive integer or "inf".
func (m *MaxOutputTokens) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if dec.PeekKind() == '"' {
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		if tok.String() == "inf" {
			*m = UnlimitedTokens
			return nil
		}
	} else if dec.PeekKind() == '0' {
		var n int64
		if err := json.UnmarshalDecode(dec, &n); err != nil {
			return err
		}
		if n > 0 {
			*m = MaxOutputTokens(n)
			return nil
		}
	}
	return errors.New("realtime: expected a positive integer or inf")
}

// Voice identifies a voice by name or by an object containing its ID. Set exactly one field.
type Voice struct {
	Name      *string
	Reference *VoiceReference
}

// VoiceReference names a voice in the object form of audio.output.voice.
type VoiceReference struct {
	ID    string                    `json:"id"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// NamedVoice constructs the string form of a voice.
func NamedVoice(name string) Voice { return Voice{Name: new(name)} }

// ReferencedVoice constructs the object form of a voice.
func ReferencedVoice(id string) Voice { return Voice{Reference: &VoiceReference{ID: id}} }

// MarshalJSONTo writes a voice in its original form.
func (v Voice) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case v.Name != nil && v.Reference == nil:
		return enc.WriteToken(jsontext.String(*v.Name))
	case v.Name == nil && v.Reference != nil:
		return json.MarshalEncode(enc, v.Reference)
	default:
		return errors.New("realtime: specify exactly one voice name or reference")
	}
}

// UnmarshalJSONFrom reads a voice name or reference.
func (v *Voice) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var out Voice
	switch dec.PeekKind() {
	case '"':
		if err := json.UnmarshalDecode(dec, &out.Name); err != nil {
			return err
		}
	case '{':
		if err := json.UnmarshalDecode(dec, &out.Reference); err != nil {
			return err
		}
	default:
		return errors.New("realtime: expected a voice name or reference")
	}
	*v = out
	return nil
}

// ClientEvent is a JSON text frame sent to the server. Audio contains PCM16 bytes, not a base64 string.
type ClientEvent struct {
	// Raw holds the entire frame for an unrecognized event type. When set, it is sent as-is instead of the typed fields.
	Raw      jsontext.Value            `json:"-"`
	Type     EventType                 `json:"type"`
	Audio    []byte                    `json:"audio,omitzero"`
	Final    *bool                     `json:"final,omitzero"`
	Session  *SessionConfig            `json:"session,omitzero"`
	Item     *Item                     `json:"item,omitzero"`
	Response *ResponseConfig           `json:"response,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// readEvent identifies a frame without interpreting members whose shape belongs to a different event type.
func readEvent(dec *jsontext.Decoder) (EventType, jsontext.Value, error) {
	raw, err := dec.ReadValue()
	if err != nil {
		return "", nil, err
	}
	var header struct {
		Type EventType `json:"type"`
	}
	if err := json.Unmarshal(raw, &header, dec.Options(), json.RejectUnknownMembers(false)); err != nil {
		return "", nil, err
	}
	return header.Type, raw, nil
}

// UnmarshalJSONFrom decodes a known client event or retains an unknown frame in Raw.
func (e *ClientEvent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	kind, raw, err := readEvent(dec)
	if err != nil {
		return err
	}
	switch kind {
	case SessionUpdate, TranscriptionSessionUpdate, InputAudioBufferAppend, InputAudioBufferCommit, InputAudioBufferClear, ConversationItemCreate, ResponseCreate, ResponseCancel:
		type wire ClientEvent
		var w wire
		if err := json.Unmarshal(raw, &w, dec.Options()); err != nil {
			return err
		}
		*e = ClientEvent(w)
	default:
		*e = ClientEvent{Type: kind, Raw: raw.Clone()}
	}
	return nil
}

// MarshalJSONTo writes the retained frame or the typed client event.
func (e ClientEvent) MarshalJSONTo(enc *jsontext.Encoder) error {
	if e.Raw != nil {
		return enc.WriteValue(e.Raw)
	}
	type wire ClientEvent
	return json.MarshalEncode(enc, wire(e))
}

// SessionConfig describes the session in a session event. Updates need only set the desired fields.
// ID, Object and Model are supplied by the server.
type SessionConfig struct {
	ID               *string                   `json:"id,omitzero"`
	Object           *string                   `json:"object,omitzero"`
	Type             *string                   `json:"type,omitzero"`
	Model            *string                   `json:"model,omitzero"`
	Instructions     *string                   `json:"instructions,omitzero"`
	OutputModalities Nullable[[]string]        `json:"output_modalities,omitzero"`
	MaxOutputTokens  *MaxOutputTokens          `json:"max_output_tokens,omitzero"`
	Tools            []Tool                    `json:"tools,omitzero"`
	ToolChoice       *string                   `json:"tool_choice,omitzero"`
	Audio            *AudioConfig              `json:"audio,omitzero"`
	Extra            map[string]jsontext.Value `json:",embed"`
}

// Tool is a realtime function tool. Parameters is the function's JSON schema.
type Tool struct {
	Type         string                    `json:"type"`
	Name         string                    `json:"name"`
	Description  *string                   `json:"description,omitzero"`
	Parameters   jsontext.Value            `json:"parameters,omitzero"`
	DeferLoading *bool                     `json:"defer_loading,omitzero"`
	Strict       *bool                     `json:"strict,omitzero"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// ResponseConfig overrides the corresponding session settings for one response.
type ResponseConfig struct {
	OutputModalities Nullable[[]string]        `json:"output_modalities,omitzero"`
	MaxOutputTokens  *MaxOutputTokens          `json:"max_output_tokens,omitzero"`
	ToolChoice       *string                   `json:"tool_choice,omitzero"`
	Extra            map[string]jsontext.Value `json:",embed"`
}

// AudioConfig describes the input and optional output audio streams.
type AudioConfig struct {
	Input  *AudioInput               `json:"input,omitzero"`
	Output *AudioOutput              `json:"output,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// AudioInput configures PCM input. The server reports null for unsupported preprocessing features.
type AudioInput struct {
	Format         *AudioFormat              `json:"format,omitzero"`
	Transcription  *Transcription            `json:"transcription,omitzero"`
	TurnDetection  jsontext.Value            `json:"turn_detection,omitzero"`
	NoiseReduction jsontext.Value            `json:"noise_reduction,omitzero"`
	Extra          map[string]jsontext.Value `json:",embed"`
}

// AudioOutput describes the generated audio stream. The server selects the format.
type AudioOutput struct {
	Format   *AudioFormat              `json:"format,omitzero"`
	Voice    Nullable[Voice]           `json:"voice,omitzero"`
	Language Nullable[string]          `json:"language,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// AudioFormat identifies PCM16 mono audio and its sample rate in Hz.
type AudioFormat struct {
	Type  *string                   `json:"type,omitzero"`
	Rate  *int                      `json:"rate,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// Transcription specifies the language and prompt for transcription. The server supplies Model.
type Transcription struct {
	Model    *string                   `json:"model,omitzero"`
	Language Nullable[string]          `json:"language,omitzero"`
	Prompt   *string                   `json:"prompt,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// Item is a conversation message, function call or function call output, identified by Type.
type Item struct {
	ID        *string                   `json:"id,omitzero"`
	Object    *string                   `json:"object,omitzero"`
	Type      string                    `json:"type"`
	Role      *string                   `json:"role,omitzero"`
	Status    *string                   `json:"status,omitzero"`
	Content   []ContentPart             `json:"content,omitzero"`
	CallID    *string                   `json:"call_id,omitzero"`
	Name      *string                   `json:"name,omitzero"`
	Arguments *string                   `json:"arguments,omitzero"`
	Output    *string                   `json:"output,omitzero"`
	Extra     map[string]jsontext.Value `json:",embed"`
}

// ContentPart is input_text, input_audio, output_text or output_audio within a message item.
type ContentPart struct {
	Type       string                    `json:"type"`
	Text       *string                   `json:"text,omitzero"`
	Transcript *string                   `json:"transcript,omitzero"`
	Extra      map[string]jsontext.Value `json:",embed"`
}

// ServerEvent is a frame received from the server. Type determines the populated fields.
// Delta carries text, partial JSON arguments, or base64 PCM16 for ResponseOutputAudioDelta.
type ServerEvent struct {
	// Raw holds the entire frame for an unrecognized event type. When set, it is encoded instead of the typed fields.
	Raw            jsontext.Value            `json:"-"`
	Type           EventType                 `json:"type"`
	EventID        string                    `json:"event_id"`
	Session        *SessionConfig            `json:"session,omitzero"`
	Item           *Item                     `json:"item,omitzero"`
	ItemID         *string                   `json:"item_id,omitzero"`
	PreviousItemID Nullable[string]          `json:"previous_item_id,omitzero"`
	Response       *Response                 `json:"response,omitzero"`
	ResponseID     *string                   `json:"response_id,omitzero"`
	OutputIndex    *int                      `json:"output_index,omitzero"`
	ContentIndex   *int                      `json:"content_index,omitzero"`
	Part           *ContentPart              `json:"part,omitzero"`
	Delta          *string                   `json:"delta,omitzero"`
	Text           *string                   `json:"text,omitzero"`
	Transcript     *string                   `json:"transcript,omitzero"`
	CallID         *string                   `json:"call_id,omitzero"`
	Name           *string                   `json:"name,omitzero"`
	Arguments      *string                   `json:"arguments,omitzero"`
	Error          *EventError               `json:"error,omitzero"`
	Extra          map[string]jsontext.Value `json:",embed"`
}

// UnmarshalJSONFrom decodes a known server event or retains an unknown frame in Raw.
func (e *ServerEvent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	kind, raw, err := readEvent(dec)
	if err != nil {
		return err
	}
	switch kind {
	case SessionCreated, SessionUpdated, InputAudioBufferCleared, InputAudioBufferCommitted,
		ConversationItemAdded, ConversationItemDone, TranscriptionDelta, TranscriptionCompleted,
		ResponseCreated, ResponseOutputItemAdded, ResponseOutputItemDone, ResponseContentPartAdded, ResponseContentPartDone,
		ResponseOutputTextDelta, ResponseOutputTextDone, ResponseOutputAudioDelta, ResponseOutputAudioDone,
		ResponseFunctionCallArgumentsDelta, ResponseFunctionCallArgumentsDone, ResponseDone, ErrorEvent:
		type wire ServerEvent
		var w wire
		if err := json.Unmarshal(raw, &w, dec.Options()); err != nil {
			return err
		}
		*e = ServerEvent(w)
	default:
		*e = ServerEvent{Type: kind, Raw: raw.Clone()}
	}
	return nil
}

// MarshalJSONTo writes the retained frame or the typed server event.
func (e ServerEvent) MarshalJSONTo(enc *jsontext.Encoder) error {
	if e.Raw != nil {
		return enc.WriteValue(e.Raw)
	}
	type wire ServerEvent
	return json.MarshalEncode(enc, wire(e))
}

// Response is the generated response in response.created or response.done.
type Response struct {
	ID               string                    `json:"id"`
	Object           string                    `json:"object"`
	Status           string                    `json:"status"`
	OutputModalities Nullable[[]string]        `json:"output_modalities,omitzero"`
	Output           []Item                    `json:"output"`
	Usage            *Usage                    `json:"usage,omitzero"`
	Extra            map[string]jsontext.Value `json:",embed"`
}

// Usage counts generated tokens in a finished response.
type Usage struct {
	OutputTokens int                       `json:"output_tokens"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// EventError is the error object of an error event; many such errors leave the session usable.
type EventError struct {
	Type    string                    `json:"type"`
	Code    *string                   `json:"code"`
	Message string                    `json:"message"`
	Param   *string                   `json:"param"`
	Extra   map[string]jsontext.Value `json:",embed"`
}
