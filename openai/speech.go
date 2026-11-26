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

	llama "github.com/zchee/go-llama.cpp"
)

// SpeechRequest generates speech in wav or pcm (signed 16-bit) format.
type SpeechRequest struct {
	llama.TTSParams
	// Voice selects a voice by name or by an object containing its id.
	Voice SpeechVoice `json:"voice,omitzero"`
	// StreamFormat selects audio streaming; the server does not support sse.
	StreamFormat *string `json:"stream_format,omitzero"`
	// Instructions is accepted but ignored by llama-server.
	Instructions *string `json:"instructions,omitzero"`
	// Extra preserves additional request fields.
	Extra map[string]jsontext.Value `json:",embed"`
}

// SpeechVoice is either a name or a voice object. Set exactly one when specifying a voice.
type SpeechVoice struct {
	// Name supplies the string form of a voice.
	Name *string
	// Object supplies the object form of a voice.
	Object *SpeechVoiceID
}

// SpeechVoiceID is an OpenAI voice identifier in object form.
type SpeechVoiceID struct {
	// ID is the model's voice name.
	ID *string `json:"id,omitzero"`
	// Extra preserves additional voice fields.
	Extra map[string]jsontext.Value `json:",embed"`
}

// MarshalJSONTo writes the selected wire form.
func (v SpeechVoice) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch {
	case v.Name != nil && v.Object == nil:
		return enc.WriteToken(jsontext.String(*v.Name))
	case v.Object != nil && v.Name == nil:
		return json.MarshalEncode(enc, v.Object)
	default:
		return errors.New("openai: speech voice requires either a name or a voice object")
	}
}

// UnmarshalJSONFrom reads either a voice name or a voice object.
func (v *SpeechVoice) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var next SpeechVoice
	switch dec.PeekKind() {
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
		return errors.New("openai: speech voice must be a string or an object")
	}
	*v = next
	return nil
}

// Speech generates audio. The caller must close the returned body or use Audio.WriteTo.
func (c *Client) Speech(ctx context.Context, req *SpeechRequest, opts ...llama.RequestOption) (*llama.Audio, error) {
	if req == nil {
		return nil, errors.New("openai: nil speech request")
	}
	if req.SpeakerRef != nil && req.Voice.Object != nil && req.Voice.Name == nil {
		// Multipart text fields arrive as strings, so send the id rather than JSON object text.
		copyReq := *req
		copyReq.Voice = SpeechVoice{Name: req.Voice.Object.ID}
		req = &copyReq
	}
	return llama.AudioCall(ctx, c.c, "/v1/audio/speech", req, req.SpeakerRef, opts...)
}
