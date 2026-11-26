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
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestSpeechWire(t *testing.T) {
	c, err := llama.New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	jsontest.CheckFallbacks[SpeechRequest](t)
	jsontest.CheckFallbacks[SpeechVoiceID](t)
	tests := map[string]struct{ fixture string }{"success: object voice and WAV": {"wav"}, "success: named voice and streaming PCM": {"pcm"}, "error: unsupported format": {"format"}}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			prefix := "testdata/speech/" + tc.fixture
			data := jsontest.ReadFile(t, prefix+".req.json")
			parsed := jsontest.RoundTrip[SpeechRequest](t, data)
			jsontest.InjectUnknown[SpeechRequest](t, data)
			jsontest.NoUnknown[SpeechRequest](t, data, nil)
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/v1/audio/speech", parsed, llama.WithModelQuery("audio"))
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			if req.Method != http.MethodPost || req.URL.Path != "/v1/audio/speech" || req.URL.RawQuery != "model=audio" || req.Header.Get("Content-Type") != "application/json" {
				t.Fatalf("unexpected request: %s %v %v", req.Method, req.URL, req.Header)
			}
			var got SpeechRequest
			if err := json.UnmarshalRead(req.Body, &got); err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(parsed, &got); diff != "" {
				t.Fatal(diff)
			}
			switch tc.fixture {
			case "wav":
				audio := jsontest.ReadFile(t, prefix+".resp.bin")
				if parsed.Voice.Object == nil || parsed.Voice.Object.ID == nil || *parsed.Voice.Object.ID != "demo" || len(audio) <= 44 || string(audio[:4]) != "RIFF" || string(audio[8:12]) != "WAVE" {
					t.Fatal("wrong object voice or WAV body")
				}
			case "pcm":
				audio := jsontest.ReadFile(t, prefix+".resp.bin")
				if parsed.Voice.Name == nil || *parsed.Voice.Name != "demo" || len(audio) == 0 || len(audio)%2 != 0 {
					t.Fatal("wrong named voice or PCM body")
				}
			case "format":
				e := llama.ReadAPIError(&http.Response{StatusCode: 400, Body: io.NopCloser(bytes.NewReader(jsontest.ReadFile(t, prefix+".resp.json")))})
				if e.Code != 400 || e.Type != llama.ErrorTypeInvalidRequest || !strings.Contains(e.Message, "response_format") {
					t.Fatal(e)
				}
			}
		})
	}
}

func FuzzSpeechVoice(f *testing.F) {
	for _, fixture := range []string{"pcm", "wav"} {
		var raw map[string]jsontext.Value
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/speech/"+fixture+".req.json"), &raw); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(raw["voice"]))
	}
	for _, s := range []string{`null`, `false`, `[]`, `{}`, `{"id":"x","future": { "x" : 1 }}`} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var v SpeechVoice
		if err := json.Unmarshal(data, &v); err != nil {
			return
		}
		opts := json.Deterministic(true)
		encoded, err := json.Marshal(v, opts)
		if err != nil {
			t.Fatal(err)
		}
		var again SpeechVoice
		if err := json.Unmarshal(encoded, &again); err != nil {
			t.Fatal(err)
		}
		out, err := json.Marshal(again, opts)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(encoded, out) {
			t.Fatalf("voice changed: %s => %s", encoded, out)
		}
	})
}

func ExampleClient_Speech() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	audio, err := NewClient(c).Speech(context.Background(), &SpeechRequest{Input: "Hello", ResponseFormat: new("wav")})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := audio.WriteTo(os.Stdout); err != nil {
		log.Fatal(err)
	}
}
