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

package openai

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"net/http"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/speech"
)

// SpeechRequest generates speech in wav or pcm (signed 16-bit) format.
type SpeechRequest struct {
	llama.TTSParams
	// Voice selects a voice by name or id object; when omitted, the server uses its configured voice unless a
	// speaker reference is provided.
	Voice SpeechVoice `json:"voice,omitzero"`
	// StreamFormat accepts "audio" to enable streaming; omission leaves the native Stream option in effect.
	StreamFormat *string `json:"stream_format,omitzero"`
	// Instructions is accepted but ignored by llama-server.
	Instructions *string `json:"instructions,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SpeechVoice is either a name or a voice object. Set at most one: the zero SpeechVoice means no voice and
// encodes as JSON null, which the server also reads as no voice.
// The realtime package uses a separate Voice for audio.output.voice, where a reference always encodes an
// id and null is represented by a wrapper; this HTTP request voice can omit the id and encode null itself.
type SpeechVoice struct {
	// Name supplies the string form of a voice.
	Name *string
	// Object supplies the object form of a voice.
	Object *SpeechVoiceID
}

// SpeechVoiceID is an OpenAI voice identifier in object form. The server reads only its id, so a decoded
// {"id": null} re-encodes as {}: the server treats a null, absent and empty id alike.
type SpeechVoiceID struct {
	// ID is the model's voice name; an absent, null, or empty id selects no named voice.
	ID *string `json:"id,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// NamedVoice constructs the string form of a speech voice.
func NamedVoice(name string) SpeechVoice { return SpeechVoice{Name: new(name)} }

// ReferencedVoice constructs the object form of a speech voice with the given ID.
func ReferencedVoice(id string) SpeechVoice { return SpeechVoice{Object: &SpeechVoiceID{ID: new(id)}} }

// MarshalJSONTo writes the selected wire form, or null for the zero SpeechVoice.
func (v SpeechVoice) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case v.Name == nil && v.Object == nil:
		return enc.WriteToken(jsontext.Null)
	case v.Name != nil && v.Object == nil:
		return enc.WriteToken(jsontext.String(*v.Name))
	case v.Object != nil && v.Name == nil:
		return json.MarshalEncode(enc, v.Object)
	default:
		return errors.New("openai: speech voice requires either a name or a voice object")
	}
}

// UnmarshalJSONFrom reads a voice name, a voice object, or null. The server accepts null as no voice, so
// null decodes as the zero SpeechVoice; a request re-encodes it as an absent member.
func (v *SpeechVoice) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var next SpeechVoice
	switch dec.PeekKind() {
	case 'n':
		if _, err := dec.ReadToken(); err != nil {
			return err
		}
	case '"':
		token, err := dec.ReadToken()
		if err != nil {
			return err
		}
		next.Name = new(token.String())
	case '{':
		next.Object = new(SpeechVoiceID)
		if err := json.UnmarshalDecode(dec, next.Object); err != nil {
			return err
		}
	default:
		return errors.New("openai: speech voice must be a string, an object or null")
	}
	*v = next
	return nil
}

// Speech generates audio. The caller must close the returned body or use Audio.WriteTo.
func (c *Client) Speech(ctx context.Context, req *SpeechRequest, opts ...llama.RequestOption) (*llama.Audio, error) {
	if req == nil {
		return nil, errors.New("openai: nil speech request")
	}
	send, ref := speechSendRequest(req)
	var body any = send
	if ref != nil {
		data, contentType, err := speech.Multipart(send, ref.Data, ref.Filename, ref.ContentType)
		if err != nil {
			return nil, c.c.RedactError(err)
		}
		body = &llama.Body{Data: data, ContentType: contentType}
	}
	resp, err := llama.CallRaw(ctx, c.c, http.MethodPost, "/v1/audio/speech", body, opts...)
	if err != nil {
		return nil, err
	}
	format, err := speech.ResponseFormat(resp)
	if err != nil {
		return nil, c.c.RedactError(err)
	}
	return &llama.Audio{Body: resp.Body, ContentType: resp.Header.Get("Content-Type"), Format: llama.AudioFormat(format)}, nil
}

// speechSendRequest selects JSON when multipart text fields would lose the type of raw_prompt.
func speechSendRequest(req *SpeechRequest) (*SpeechRequest, *llama.SpeakerRef) {
	ref := req.SpeakerRef
	if ref == nil || req.RawPrompt == nil {
		return req, ref
	}
	r := *req
	r.SpeakerRefB64 = ref.Data
	r.SpeakerRef = nil
	return &r, nil
}
