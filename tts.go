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
	"context"
	"encoding/json/jsontext"
	"errors"
	"net/http"

	"github.com/zchee/go-llama.cpp/internal/speech"
)

// TTSParams holds the synthesis options shared by the native and OpenAI speech endpoints.
type TTSParams struct {
	// Input is the text to speak. Prompt is its alias when Input is nil; a present but empty Input is
	// rejected by the server even when Prompt is set, so presence is preserved.
	Input *string `json:"input,omitzero"`
	// Prompt supplies the text when Input is absent.
	Prompt *string `json:"prompt,omitzero"`
	// Model selects the model in router mode.
	Model *string `json:"model,omitzero"`
	// Lang selects the language on models with language controls.
	Lang *string `json:"lang,omitzero"`
	// RawPrompt sends the text to the model as is (for example phonemes), skipping the model's "tts" chat
	// template.
	RawPrompt *bool `json:"raw_prompt,omitzero"`
	// SpeakerRef is uploaded as a multipart file and takes precedence over SpeakerRefB64. When RawPrompt is
	// also set, the request is sent as JSON carrying the reference bytes in SpeakerRefB64 instead: the
	// server's multipart parser restores the types of its numeric and stream fields but leaves raw_prompt a
	// string, which the handler silently reads as false.
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
	// Stream requests an audio byte stream, not server-sent events. A generation that fails after the
	// response status has been sent ends the stream with a clean EOF, which the client cannot tell from a
	// complete stream.
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
	send, ref := ttsSendRequest(req)
	var body any = send
	if ref != nil {
		data, contentType, err := speech.Multipart(send, ref.Data, ref.Filename, ref.ContentType)
		if err != nil {
			return nil, c.RedactError(err)
		}
		body = &Body{Data: data, ContentType: contentType}
	}
	resp, err := CallRaw(ctx, c, http.MethodPost, "/tts", body, opts...)
	if err != nil {
		return nil, err
	}
	format, err := speech.ResponseFormat(resp)
	if err != nil {
		return nil, c.RedactError(err)
	}
	return &Audio{Body: resp.Body, ContentType: resp.Header.Get("Content-Type"), Format: AudioFormat(format)}, nil
}

// ttsSendRequest returns the request to send and the reference file to upload. With both an uploaded
// reference and RawPrompt set, the request is sent as JSON carrying the reference in SpeakerRefB64, because
// a multipart form cannot carry the boolean: the server reads a form raw_prompt field as a string and
// silently falls back to false. The uploaded bytes replace any SpeakerRefB64 value, keeping the documented
// precedence.
func ttsSendRequest(req *TTSRequest) (*TTSRequest, *SpeakerRef) {
	ref := req.SpeakerRef
	if ref == nil || req.RawPrompt == nil {
		return req, ref
	}
	r := *req
	r.SpeakerRefB64 = ref.Data
	r.SpeakerRef = nil
	return &r, nil
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
