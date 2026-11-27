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
	// Input is the text to speak. When it is nil, the server uses Prompt. The server rejects an explicitly empty
	// Input even when Prompt is set, so the client sends a non-nil empty Input as is.
	Input *string `json:"input,omitzero"`
	// Prompt supplies the text when Input is absent.
	Prompt *string `json:"prompt,omitzero"`
	// Model is required in router mode to select the model; the client omits an empty name.
	Model string `json:"model,omitzero"`
	// Lang selects a model-dependent language code and uses the server's configured language when omitted.
	Lang *string `json:"lang,omitzero"`
	// RawPrompt bypasses the "tts" template to send model-ready text such as phonemes; it defaults to false.
	RawPrompt *bool `json:"raw_prompt,omitzero"`
	// SpeakerRef takes precedence over SpeakerRefB64. The client uploads it as a multipart file unless RawPrompt
	// is non-nil, in which case it sends the bytes as SpeakerRefB64 in JSON. Unlike numeric and stream fields,
	// multipart raw_prompt remains a string that the server silently reads as false.
	SpeakerRef *SpeakerRef `json:"-"`
	// SpeakerRefB64 supplies optional reference audio for voice cloning, encoded as base64 in JSON.
	SpeakerRefB64 []byte `json:"speaker_ref_b64,omitzero"`
	// SpeakerRefText supplies an optional transcript of the reference audio for models that use it in the prompt.
	SpeakerRefText *string `json:"speaker_ref_text,omitzero"`
	// TopK caps candidates for both samplers; zero or nil keeps the server's backbone and model's audio defaults.
	TopK *int `json:"top_k,omitzero"`
	// TopP caps cumulative sampling probability in [0, 1], with zero or nil keeping the sampling defaults.
	TopP *float64 `json:"top_p,omitzero"`
	// Temperature is nonnegative: zero selects greedy sampling and nil keeps the backbone and audio defaults.
	Temperature *float64 `json:"temperature,omitzero"`
	// RepeatPenalty discourages backbone repetition over the utterance and defaults to 1.05.
	RepeatPenalty *float64 `json:"repeat_penalty,omitzero"`
	// Seed seeds both the backbone and audio generator; -1 requests randomness and nil uses the server seed.
	Seed *int64 `json:"seed,omitzero"`
	// Speed is the positive pace multiplier for models that support it and defaults to 1.0, the model's own pace.
	Speed *float64 `json:"speed,omitzero"`
	// Loudness sets a negative target in LUFS on supported models and defaults to zero, the model's own loudness.
	Loudness *float64 `json:"loudness,omitzero"`
	// CFGScale sets classifier-free guidance: one disables it and zero or nil uses the model default.
	CFGScale *float64 `json:"cfg_scale,omitzero"`
	// FlowSteps sets 1 to 1024 solver steps on models with flow matching; zero or nil uses the model default.
	FlowSteps *int `json:"flow_steps,omitzero"`
	// NPredict caps the number of generated audio frames, with -1 or omission using the model's limit.
	NPredict *int `json:"n_predict,omitzero"`
	// MaxTokens is an alias for NPredict, which takes precedence if both are set.
	MaxTokens *int `json:"max_tokens,omitzero"`
	// Stream requests an audio byte stream rather than a buffered response and defaults to false.
	// A failure after the status is sent ends the stream with a clean EOF indistinguishable from completion.
	Stream *bool `json:"stream,omitzero"`
	// ResponseFormat selects the encoding, defaulting to WAV; the meaning of "pcm" depends on the endpoint.
	ResponseFormat *string `json:"response_format,omitzero"`
}

// TTSRequest requests native speech synthesis in wav, pcm (float32), or pcm16 format.
type TTSRequest struct {
	TTSParams
	// Voice names a predefined voice; without a speaker reference, omission uses the configured or model default.
	Voice *string `json:"voice,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
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
