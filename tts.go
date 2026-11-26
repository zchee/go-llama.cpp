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
	"cmp"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"maps"
	"mime"
	"mime/multipart"
	"net/textproto"
	"slices"
	"strings"
)

// TTSParams holds the synthesis options shared by the native and OpenAI speech endpoints.
type TTSParams struct {
	// Input is the text to speak. Prompt is its alias when Input is absent.
	Input string `json:"input,omitzero"`
	// Prompt supplies the text when Input is absent.
	Prompt *string `json:"prompt,omitzero"`
	// Model selects the model in router mode.
	Model *string `json:"model,omitzero"`
	// Lang selects the language on models with language controls.
	Lang *string `json:"lang,omitzero"`
	// RawPrompt bypasses the model's text preprocessing when sent as JSON.
	RawPrompt *bool `json:"raw_prompt,omitzero"`
	// SpeakerRef is uploaded as a multipart file and takes precedence over SpeakerRefB64.
	SpeakerRef *SpeakerRef `json:"-"`
	// SpeakerRefB64 holds reference audio bytes; JSON encodes them as base64.
	SpeakerRefB64 []byte `json:"speaker_ref_b64,omitzero"`
	// SpeakerRefText is the transcript of the reference audio.
	SpeakerRefText *string `json:"speaker_ref_text,omitzero"`
	// TopK limits the number of candidates in audio and backbone sampling.
	TopK *int `json:"top_k,omitzero"`
	// TopP limits the cumulative probability of candidates.
	TopP *float64 `json:"top_p,omitzero"`
	// Temperature controls sampling randomness; zero selects greedy sampling.
	Temperature *float64 `json:"temperature,omitzero"`
	// RepeatPenalty discourages repetition over the generated utterance.
	RepeatPenalty *float64 `json:"repeat_penalty,omitzero"`
	// Seed sets the RNG seed for both samplers; -1 requests a random seed.
	Seed *int64 `json:"seed,omitzero"`
	// Speed sets the relative speaking speed on models supporting it.
	Speed *float64 `json:"speed,omitzero"`
	// Loudness sets the target loudness in LUFS on models supporting it.
	Loudness *float64 `json:"loudness,omitzero"`
	// CFGScale sets classifier-free guidance on models supporting it.
	CFGScale *float64 `json:"cfg_scale,omitzero"`
	// FlowSteps sets the number of flow-matching solver steps on models supporting it.
	FlowSteps *int `json:"flow_steps,omitzero"`
	// NPredict limits the number of audio frames to generate.
	NPredict *int `json:"n_predict,omitzero"`
	// MaxTokens is an alias for NPredict.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// Stream requests an audio byte stream, not server-sent events.
	Stream *bool `json:"stream,omitzero"`
	// ResponseFormat selects the encoding; the meaning of pcm depends on the endpoint.
	ResponseFormat *string `json:"response_format,omitzero"`
}

// TTSRequest requests native speech synthesis in wav, pcm (float32), or pcm16 format.
type TTSRequest struct {
	TTSParams
	// Voice selects a voice by name from the model's predefined voices.
	Voice *string `json:"voice,omitzero"`
	// Extra preserves additional request fields.
	Extra map[string]jsontext.Value `json:",embed"`
}

// TTS generates speech. The caller must close the returned audio body or use Audio.WriteTo.
func (c *Client) TTS(ctx context.Context, req *TTSRequest, opts ...RequestOption) (*Audio, error) {
	if req == nil {
		return nil, errors.New("llama: nil TTS request")
	}
	return AudioCall(ctx, c, "/tts", req, req.SpeakerRef, opts...)
}

// SpeakerRef holds a reference audio file in memory so an HTTP retry can resend it.
type SpeakerRef struct {
	// Data is the complete encoded reference file.
	Data []byte
	// Filename defaults to reference.wav.
	Filename string
	// ContentType defaults to application/octet-stream.
	ContentType string
}

func (r *SpeakerRef) ttsMultipart(request any) (*Body, error) {
	data, err := json.Marshal(request, jsonOptions)
	if err != nil {
		return nil, err
	}
	var fields map[string]jsontext.Value
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, k := range slices.Sorted(maps.Keys(fields)) {
		v := fields[k]
		text := string(v)
		if v.Kind() == '"' {
			if err := json.Unmarshal(v, &text); err != nil {
				return nil, err
			}
		}
		if strings.ContainsAny(k, "\r\n") {
			return nil, errors.New("llama: invalid speech form field name")
		}
		if err := w.WriteField(k, text); err != nil {
			return nil, err
		}
	}
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{"name": "speaker_ref", "filename": cmp.Or(r.Filename, "reference.wav")}))
	if strings.ContainsAny(r.ContentType, "\r\n") {
		return nil, errors.New("llama: invalid speaker reference content type")
	}
	header.Set("Content-Type", cmp.Or(r.ContentType, "application/octet-stream"))
	part, err := w.CreatePart(header)
	if err != nil {
		return nil, err
	}
	if _, err := part.Write(r.Data); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return &Body{ContentType: w.FormDataContentType(), Data: buf.Bytes()}, nil
}
