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
	// Value holds a non-null member; leave it nil for null or an omitted member.
	Value *T
	// Null records an explicit JSON null, distinct from omission when Value is nil.
	Null bool
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
// The openai package uses a separate SpeechVoice for a top-level HTTP request member that can omit the
// id and encode null directly, whereas realtime always encodes a reference id and wraps Voice for null.
type Voice struct {
	// Name selects a model voice as a JSON string instead of an ID object.
	Name *string
	// Reference selects a model voice using an object with an id member instead of a string.
	Reference *VoiceReference
}

// VoiceReference names a voice in the object form of audio.output.voice.
type VoiceReference struct {
	// ID matches a voice name from the model's voices list in GET /props.
	ID string `json:"id"`
	// Extra holds members this type does not declare; they survive a round trip.
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
	Raw jsontext.Value `json:"-"`
	// Type selects the server action taken for this frame.
	Type EventType `json:"type"`
	// Audio carries mono PCM16 bytes in input_audio_buffer.append; omitted chunks append nothing.
	Audio []byte `json:"audio,omitzero"`
	// Final ends live transcription after pending audio drains when true on input_audio_buffer.commit;
	// omission makes a normal commit.
	Final *bool `json:"final,omitzero"`
	// Session carries the settings to change on session.update or transcription_session.update.
	Session *SessionConfig `json:"session,omitzero"`
	// Item carries a user text message or a tool result on conversation.item.create.
	Item *Item `json:"item,omitzero"`
	// Response carries optional one-turn overrides on response.create.
	Response *ResponseConfig `json:"response,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
	// ID is assigned by the server at handshake and stays the same across updates.
	ID *string `json:"id,omitzero"`
	// Object is "realtime.session" for conversation or "realtime.transcription_session" for transcription.
	Object *string `json:"object,omitzero"`
	// Type selects "realtime" or "transcription"; turn-based sessions can switch only before audio or messages.
	Type *string `json:"type,omitzero"`
	// Model names the loaded model in a conversation snapshot; transcription reports it under Audio instead.
	Model *string `json:"model,omitzero"`
	// Instructions sets the system prompt before the first audio or message; omission leaves it unchanged.
	Instructions *string `json:"instructions,omitzero"`
	// OutputModalities selects text, audio or both for responses; null resets automatic selection, the default.
	OutputModalities Nullable[[]string] `json:"output_modalities,omitzero"`
	// MaxOutputTokens defaults to UnlimitedTokens; positive limits above 2147483647 are capped there.
	MaxOutputTokens *MaxOutputTokens `json:"max_output_tokens,omitzero"`
	// Tools replaces the available functions before the first audio or message; an empty list clears them.
	Tools []Tool `json:"tools,omitzero"`
	// ToolChoice selects "auto" (the default), "none" or "required" for responses.
	ToolChoice *string `json:"tool_choice,omitzero"`
	// Audio configures the PCM input and optional speech output; omission leaves both unchanged.
	Audio *AudioConfig `json:"audio,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Tool is a realtime function tool. Parameters is the function's JSON schema.
type Tool struct {
	// Type must be "function" for the server to use the tool; it skips other values.
	Type string `json:"type"`
	// Name is the required function name reported in tool calls.
	Name string `json:"name"`
	// Description optionally tells the model when to use the function.
	Description *string `json:"description,omitzero"`
	// Parameters describes function arguments as JSON Schema; omission uses an empty object.
	Parameters jsontext.Value `json:"parameters,omitzero"`
	// DeferLoading marks this function as deferred in the chat template; omission defaults to false.
	DeferLoading *bool `json:"defer_loading,omitzero"`
	// Strict is preserved in session snapshots but does not change argument generation.
	Strict *bool `json:"strict,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ResponseConfig overrides the corresponding session settings for one response.
type ResponseConfig struct {
	// OutputModalities overrides text or audio selection for this turn; null uses automatic selection.
	OutputModalities Nullable[[]string] `json:"output_modalities,omitzero"`
	// MaxOutputTokens overrides the session cap with a positive count or UnlimitedTokens.
	MaxOutputTokens *MaxOutputTokens `json:"max_output_tokens,omitzero"`
	// ToolChoice overrides the session mode with "auto", "none" or "required" for this turn.
	ToolChoice *string `json:"tool_choice,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AudioConfig describes the input and optional output audio streams.
type AudioConfig struct {
	// Input describes received PCM audio; omitting it on update retains the current input settings.
	Input *AudioInput `json:"input,omitzero"`
	// Output describes generated PCM audio and is omitted in snapshots without audio output.
	Output *AudioOutput `json:"output,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AudioInput configures PCM input. The server reports null for unsupported preprocessing features.
type AudioInput struct {
	// Format reports or sets the PCM16 mono input sample rate; omitted updates leave it unchanged.
	Format *AudioFormat `json:"format,omitzero"`
	// Transcription reports transcription settings and allows updates in turn-based transcription sessions.
	Transcription *Transcription `json:"transcription,omitzero"`
	// TurnDetection is reported as null because the server does not use automatic turn detection for this API.
	TurnDetection jsontext.Value `json:"turn_detection,omitzero"`
	// NoiseReduction is reported as null in live transcription snapshots and omitted otherwise.
	NoiseReduction jsontext.Value `json:"noise_reduction,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AudioOutput describes the generated audio stream. The server selects the format.
type AudioOutput struct {
	// Format reports the model's mono PCM16 output sample rate; settings updates cannot override it.
	Format *AudioFormat `json:"format,omitzero"`
	// Voice selects a model voice before the first audio or message; null clears the configured voice.
	Voice Nullable[Voice] `json:"voice,omitzero"`
	// Language selects speech language before the first audio or message; null and omission leave it unchanged.
	Language Nullable[string] `json:"language,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// AudioFormat identifies PCM16 mono audio and its sample rate in Hz.
type AudioFormat struct {
	// Type is reported as "audio/pcm" for mono PCM16 and is not read from updates.
	Type *string `json:"type,omitzero"`
	// Rate is the sample rate in Hz, defaulting to the model rate; input updates accept 8000 to 192000 Hz.
	Rate *int `json:"rate,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Transcription specifies the language and prompt for transcription. The server supplies Model.
type Transcription struct {
	// Model names the loaded transcription model in session snapshots and is not read from updates.
	Model *string `json:"model,omitzero"`
	// Language sets the language hint for turn-based transcription; null and omission leave the hint unchanged.
	Language Nullable[string] `json:"language,omitzero"`
	// Prompt overrides the default turn-based transcription instruction when supported by the model.
	Prompt *string `json:"prompt,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Item is a conversation message, function call or function call output, identified by Type.
type Item struct {
	// ID is echoed on item creation if supplied; otherwise the server assigns it.
	ID *string `json:"id,omitzero"`
	// Object is "realtime.item" on items emitted by the server.
	Object *string `json:"object,omitzero"`
	// Type selects "message", "function_call" or "function_call_output"; only the server sends function calls.
	Type string `json:"type"`
	// Role is required to be "user" for a submitted message and is omitted on tool calls and results.
	Role *string `json:"role,omitzero"`
	// Status is "completed" for caller items and "in_progress", "completed" or "cancelled" for generated items.
	Status *string `json:"status,omitzero"`
	// Content contains message parts; item creation accepts only input_text parts, which the server concatenates.
	Content []ContentPart `json:"content,omitzero"`
	// CallID links a function_call_output to a function_call from the last response.
	CallID *string `json:"call_id,omitzero"`
	// Name identifies the function on a generated function_call item, not its result.
	Name *string `json:"name,omitzero"`
	// Arguments is the tool call's JSON argument text, initially empty in response.output_item.added.
	Arguments *string `json:"arguments,omitzero"`
	// Output supplies the function result as text, defaulting to an empty string on function_call_output.
	Output *string `json:"output,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ContentPart is input_text, input_audio, output_text or output_audio within a message item.
type ContentPart struct {
	// Type describes the direction and modality; only the server supplies values other than "input_text".
	Type string `json:"type"`
	// Text carries input or output text, empty when omitted on input or when an output part first opens.
	Text *string `json:"text,omitzero"`
	// Transcript is empty on output_audio parts; generated text is emitted separately as output_text.
	Transcript *string `json:"transcript,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ServerEvent is a frame received from the server. Type determines the populated fields.
// Delta carries text, partial JSON arguments, or base64 PCM16 for ResponseOutputAudioDelta.
type ServerEvent struct {
	// Raw holds the entire frame for an unrecognized event type. When set, it is encoded instead of the typed fields.
	Raw jsontext.Value `json:"-"`
	// Type identifies the server event and determines which payload members are present.
	Type EventType `json:"type"`
	// EventID is assigned independently by the server for each event frame.
	EventID string `json:"event_id"`
	// Session is the configuration snapshot in session.created and session.updated.
	Session *SessionConfig `json:"session,omitzero"`
	// Item describes the item on [ConversationItemAdded], [ConversationItemDone],
	// [ResponseOutputItemAdded] and [ResponseOutputItemDone].
	Item *Item `json:"item,omitzero"`
	// ItemID links audio commits, transcription, response content parts, text/audio output and tool-argument events
	// to the corresponding item.
	ItemID *string `json:"item_id,omitzero"`
	// PreviousItemID links input_audio_buffer.committed, conversation.item.added and conversation.item.done
	// to the preceding item, or is null for the first item.
	PreviousItemID Nullable[string] `json:"previous_item_id,omitzero"`
	// Response is the response snapshot on response.created and response.done.
	Response *Response `json:"response,omitzero"`
	// ResponseID ties output-item, content-part, text/audio output and tool-argument events to their response.
	ResponseID *string `json:"response_id,omitzero"`
	// OutputIndex gives the zero-based position in Response.Output on events that carry ResponseID.
	OutputIndex *int `json:"output_index,omitzero"`
	// ContentIndex gives the zero-based part index on response content-part and text/audio output events and is 0
	// on input audio transcription events.
	ContentIndex *int `json:"content_index,omitzero"`
	// Part is the content snapshot on response.content_part.added and response.content_part.done.
	Part *ContentPart `json:"part,omitzero"`
	// Delta carries transcript text, output text, base64 PCM16 or partial tool JSON in the matching delta event.
	Delta *string `json:"delta,omitzero"`
	// Text is the full generated text on response.output_text.done.
	Text *string `json:"text,omitzero"`
	// Transcript is the full text of the input item on conversation.item.input_audio_transcription.completed.
	Transcript *string `json:"transcript,omitzero"`
	// CallID identifies the call on [ResponseFunctionCallArgumentsDelta] and [ResponseFunctionCallArgumentsDone].
	CallID *string `json:"call_id,omitzero"`
	// Name identifies the called function on response.function_call_arguments.done.
	Name *string `json:"name,omitzero"`
	// Arguments is the complete argument string on response.function_call_arguments.done.
	Arguments *string `json:"arguments,omitzero"`
	// Error is the diagnostic object on an error event.
	Error *EventError `json:"error,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
	// ID is assigned when the response opens and stays the same through response.done.
	ID string `json:"id"`
	// Object is "realtime.response" in both snapshots.
	Object string `json:"object"`
	// Status is "in_progress" on response.created and "completed" or "cancelled" on response.done.
	Status string `json:"status"`
	// OutputModalities is the chosen mode in response.created, null for automatic, and omitted in response.done.
	OutputModalities Nullable[[]string] `json:"output_modalities,omitzero"`
	// Output is empty on response.created and contains the generated items in order on response.done.
	Output []Item `json:"output"`
	// Usage reports the total generated tokens only on response.done.
	Usage *Usage `json:"usage,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Usage counts generated tokens in a finished response.
type Usage struct {
	// OutputTokens counts tokens decoded for this response, excluding prompt tokens.
	OutputTokens int `json:"output_tokens"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// EventError is the error object of an error event; many such errors leave the session usable.
type EventError struct {
	// Type carries the server reason, such as "invalid_request_error" or "conversation_empty".
	Type string `json:"type"`
	// Code is always null; the server reports the reason in Type instead.
	Code *string `json:"code"`
	// Message describes the problem in text for the caller.
	Message string `json:"message"`
	// Param is always null because the server does not identify a specific parameter here.
	Param *string `json:"param"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}
