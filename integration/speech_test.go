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

package integration

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/openai"
)

var presetSpeech = &serverPreset{name: "tinyPocketTTS", options: func(t testing.TB) harness.Options {
	return harness.Options{Model: harness.Model(t, "tinyPocketTTS"), Args: []string{"--mmproj", harness.Model(t, "tinyPocketTTS-mmproj"), "--parallel", "1", "--ctx-size", "512"}}
}}

func TestSpeechAudio(t *testing.T) {
	c, obs := presetSpeech.server(t).Client(t)
	wav := jsontest.ReadFile(t, "../testdata/tts/wav.resp.bin")
	ref := &llama.SpeakerRef{Data: wav, Filename: "reference.wav", ContentType: "audio/wav"}
	wavFormat := llama.AudioFormat{MediaType: "audio/wav"}
	f32 := llama.AudioFormat{MediaType: "audio/pcm", SampleRate: 24000, Encoding: "float", Bits: 32, Channels: 1}
	s16 := llama.AudioFormat{MediaType: "audio/pcm", SampleRate: 24000, Encoding: "signed-int", Bits: 16, Channels: 1}
	tests := map[string]struct {
		native        *llama.TTSRequest
		oai           *openai.SpeechRequest
		format        llama.AudioFormat
		streamWAV     bool
		errorContains string
	}{
		"success: streaming WAV with unknown length":           {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), Stream: new(true)}, format: wavFormat, streamWAV: true},
		"success: streaming float PCM":                         {native: &llama.TTSRequest{Prompt: new("Hi"), MaxTokens: new(1), Stream: new(true), ResponseFormat: new("pcm")}, format: f32},
		"success: signed PCM":                                  {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), ResponseFormat: new("pcm16")}, format: s16},
		"success: native unknown format falls back to WAV":     {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), ResponseFormat: new("other")}, format: wavFormat},
		"success: valid base64 reference":                      {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), SpeakerRefB64: wav, SpeakerRefText: new("Hi")}, format: wavFormat},
		"success: uploaded reference takes precedence":         {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), SpeakerRef: ref, SpeakerRefB64: []byte("not audio")}, format: wavFormat},
		"success: speech object voice with file":               {oai: &openai.SpeechRequest{Input: "Hi", NPredict: new(1), Voice: openai.SpeechVoice{Object: &openai.SpeechVoiceID{ID: new("demo")}}, SpeakerRef: ref}, format: wavFormat},
		"success: speech named voice and streaming signed PCM": {oai: &openai.SpeechRequest{Input: "Hi", NPredict: new(1), Voice: openai.SpeechVoice{Name: new("demo")}, ResponseFormat: new("pcm"), StreamFormat: new("audio")}, format: s16},
		"error: unsupported speed":                             {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), Speed: new(1.5)}, errorContains: "speed"},
		"error: unsupported loudness":                          {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), Loudness: new(-16.)}, errorContains: "loudness"},
		"error: unsupported guidance":                          {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), CFGScale: new(2.)}, errorContains: "cfg_scale"},
		"error: unsupported flow steps":                        {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), FlowSteps: new(1)}, errorContains: "flow_steps"},
		"error: invalid audio reference":                       {native: &llama.TTSRequest{Input: "Hi", NPredict: new(1), SpeakerRef: &llama.SpeakerRef{Data: []byte("not audio")}}, errorContains: "speaker_ref"},
		"error: speech does not accept pcm16":                  {oai: &openai.SpeechRequest{Input: "Hi", ResponseFormat: new("pcm16")}, errorContains: "response_format"},
		"error: speech does not stream SSE":                    {oai: &openai.SpeechRequest{Input: "Hi", StreamFormat: new("sse")}, errorContains: "stream_format"},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			obs.Reset()
			var a *llama.Audio
			var err error
			if tc.native != nil {
				a, err = c.TTS(t.Context(), tc.native)
			} else {
				a, err = openai.NewClient(c).Speech(t.Context(), tc.oai)
			}
			if tc.errorContains != "" {
				e, ok := errors.AsType[*llama.APIError](err)
				if a != nil || !ok || e.StatusCode != 400 || !strings.Contains(e.Message, tc.errorContains) {
					t.Fatalf("got audio %v, error %v; want 400 mentioning %q", a != nil, err, tc.errorContains)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			ex := obs.Exchanges()
			if len(ex) != 1 || ex[0].BodyClosed() {
				_ = a.Body.Close()
				t.Fatal("caller did not receive one open body")
			}
			var data bytes.Buffer
			n, err := a.WriteTo(&data)
			if err != nil {
				t.Fatal(err)
			}
			if !ex[0].BodyClosed() || n != int64(data.Len()) || n == 0 || ex[0].BytesRead() != n {
				t.Fatalf("read %d bytes, copied %d, closed=%t", ex[0].BytesRead(), n, ex[0].BodyClosed())
			}
			if a.Format != tc.format {
				t.Errorf("Format = %+v, want %+v", a.Format, tc.format)
			}
			if !strings.HasPrefix(a.ContentType, a.Format.MediaType) {
				t.Fatalf("unexpected Content-Type %q", a.ContentType)
			}
			if a.Format.MediaType == "audio/wav" {
				b := data.Bytes()
				if len(b) <= 44 || string(b[:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
					t.Fatal("invalid WAV")
				}
				if (binary.LittleEndian.Uint32(b[4:8]) == math.MaxUint32) != tc.streamWAV {
					t.Fatalf("unexpected RIFF size %x", b[4:8])
				}
			} else if n%int64(a.Format.Channels*a.Format.Bits/8) != 0 {
				t.Fatalf("incomplete PCM frame: %d bytes", n)
			}
		})
	}
}

func TestSpeechBodyOwnership(t *testing.T) {
	c, obs := presetSpeech.server(t).Client(t)
	tests := map[string]struct{ write bool }{"success: caller closes unread body": {}, "error: failing writer still closes body": {true}}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			obs.Reset()
			a, err := c.TTS(t.Context(), &llama.TTSRequest{Input: "Hi", NPredict: new(1), Stream: new(true)})
			if err != nil {
				t.Fatal(err)
			}
			if tc.write {
				reader, writer := io.Pipe()
				_ = reader.Close()
				defer writer.Close()
				n, err := a.WriteTo(writer)
				if n != 0 || !errors.Is(err, io.ErrClosedPipe) {
					t.Fatalf("WriteTo = %d, %v", n, err)
				}
			} else if err := a.Body.Close(); err != nil {
				t.Fatal(err)
			}
			if ex := obs.Exchanges(); len(ex) != 1 || !ex[0].BodyClosed() {
				t.Fatal("response body not closed")
			}
		})
	}
}

func TestSpeechAlias(t *testing.T) {
	c, _ := presetSpeech.server(t).Client(t)
	r, err := llama.CallRaw(t.Context(), c, http.MethodPost, "/audio/speech", &openai.SpeechRequest{Input: "Hi", NPredict: new(1)})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	var magic [4]byte
	if _, err := io.ReadFull(r.Body, magic[:]); err != nil {
		t.Fatal(err)
	}
	if string(magic[:]) != "RIFF" || r.Header.Get("Content-Type") != "audio/wav" {
		t.Fatal("alias did not return WAV audio")
	}
}
