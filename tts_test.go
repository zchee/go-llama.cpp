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
	"context"
	"encoding/binary"
	json "encoding/json/v2"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestTTSWire(t *testing.T) {
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	jsontest.CheckFallbacks[TTSRequest](t)
	tests := map[string]struct {
		fixture, message string
		sampleBytes      int
	}{
		"success: WAV":           {"wav", "", 0},
		"success: streaming WAV": {"stream_wav", "", 0},
		"success: streaming float PCM with prompt and frame aliases": {"pcm", "", 4},
		"success: signed PCM":          {"pcm16", "", 2},
		"error: undecodable reference": {"reference", "speaker_ref", 0},
		"error: speed unsupported":     {"speed", "speed", 0},
		"error: loudness unsupported":  {"loudness", "loudness", 0},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/tts/" + tc.fixture
			data := jsontest.ReadFile(t, prefix+".req.json")
			parsed := jsontest.RoundTrip[TTSRequest](t, data)
			jsontest.NoUnknown[TTSRequest](t, data, nil)
			jsontest.InjectUnknown[TTSRequest](t, data)
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/tts", parsed, WithModelQuery("audio"))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != http.MethodPost || req.URL.Path != "/tts" || req.URL.RawQuery != "model=audio" || req.Header.Get("Content-Type") != "application/json" {
				t.Fatalf("unexpected request: %s %v %v", req.Method, req.URL, req.Header)
			}
			var got TTSRequest
			if err := json.UnmarshalRead(req.Body, &got); err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(parsed, &got); diff != "" {
				t.Fatal(diff)
			}
			if tc.message != "" {
				data := jsontest.ReadFile(t, prefix+".resp.json")
				e := ReadAPIError(&http.Response{StatusCode: 400, Body: io.NopCloser(bytes.NewReader(data))})
				if e.Code != 400 || e.Type != ErrorTypeInvalidRequest || !strings.Contains(e.Message, tc.message) {
					t.Fatal(e)
				}
				return
			}
			audio := jsontest.ReadFile(t, prefix+".resp.bin")
			if tc.sampleBytes != 0 {
				if len(audio) == 0 || len(audio)%tc.sampleBytes != 0 {
					t.Fatalf("invalid PCM byte length %d", len(audio))
				}
				return
			}
			if len(audio) <= 44 || string(audio[:4]) != "RIFF" || string(audio[8:12]) != "WAVE" {
				t.Fatal("missing WAV header or samples")
			}
			size := int64(binary.LittleEndian.Uint32(audio[4:8]))
			want := int64(len(audio) - 8)
			if parsed.Stream != nil && *parsed.Stream {
				want = math.MaxUint32
			}
			if size != want {
				t.Errorf("RIFF size = %d, want %d", size, want)
			}
		})
	}
}

func TestTTSMultipartReplay(t *testing.T) {
	wav := jsontest.ReadFile(t, "testdata/tts/wav.resp.bin")
	recorded := jsontest.ReadFile(t, "testdata/tts/multipart.req.bin")
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	r := &TTSRequest{Input: "Hi", NPredict: new(1), Seed: new(int64(42)), ResponseFormat: new("wav"), SpeakerRef: &SpeakerRef{Data: wav, Filename: "reference.wav", ContentType: "audio/wav"}}
	form, err := r.SpeakerRef.ttsMultipart(r)
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct{ body *Body }{
		"success: recorded form":  {&Body{ContentType: "multipart/form-data; boundary=speech-reference", Data: recorded}},
		"success: generated form": {form},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/tts", tc.body)
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			replay, err := req.GetBody()
			if err != nil {
				t.Fatal(err)
			}
			defer replay.Close()
			data, err := io.ReadAll(replay)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, tc.body.Data) {
				t.Fatal("replayed multipart differs from original body")
			}
			if err := req.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := req.MultipartForm.RemoveAll(); err != nil {
					t.Error(err)
				}
			})
			want := map[string][]string{"input": {"Hi"}, "n_predict": {"1"}, "response_format": {"wav"}, "seed": {"42"}}
			if diff := gocmp.Diff(want, req.MultipartForm.Value); diff != "" {
				t.Fatal(diff)
			}
			f, h, err := req.FormFile("speaker_ref")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			got, err := io.ReadAll(f)
			if err != nil {
				t.Fatal(err)
			}
			if h.Filename != "reference.wav" || h.Header.Get("Content-Type") != "audio/wav" || !bytes.Equal(got, wav) {
				t.Fatal("speaker reference metadata or bytes differ")
			}
		})
	}
}

func ExampleClient_TTS() {
	c, err := New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	audio, err := c.TTS(context.Background(), &TTSRequest{Input: "Hello"})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := audio.WriteTo(os.Stdout); err != nil {
		log.Fatal(err)
	}
}
