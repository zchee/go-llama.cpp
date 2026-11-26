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
	"encoding/base64"
	"encoding/binary"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"io"
	"log"
	"math"
	"mime"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/internal/speech"
)

func TestTTSRequestModel(t *testing.T) {
	tests := map[string]struct {
		input string
		want  string
	}{
		"success: empty model omitted": {input: `{"model":""}`, want: `{}`},
		"success: explicit model kept": {input: `{"model":"audio"}`, want: `{"model":"audio"}`},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var req TTSRequest
			if err := json.Unmarshal([]byte(tc.input), &req); err != nil {
				t.Fatal(err)
			}
			got, err := json.Marshal(req)
			if err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tc.want, string(got)); diff != "" {
				t.Fatalf("request JSON mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

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
		"success: signed PCM":                          {"pcm16", "", 2},
		"success: raw prompt with base64 reference":    {"raw_b64", "", 0},
		"error: empty input is not replaced by prompt": {"empty_input", "input", 0},
		"error: undecodable reference":                 {"reference", "speaker_ref", 0},
		"error: speed unsupported":                     {"speed", "speed", 0},
		"error: loudness unsupported":                  {"loudness", "loudness", 0},
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
	response := jsontest.ReadFile(t, "testdata/tts/multipart.resp.bin")
	if len(response) <= 44 || string(response[:4]) != "RIFF" || string(response[8:12]) != "WAVE" {
		t.Fatal("the server did not answer the recorded multipart request with WAV audio")
	}
	c, err := New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	r := &TTSRequest{Input: new("Hi"), NPredict: new(1), Seed: new(int64(42)), ResponseFormat: new("wav"), SpeakerRef: &SpeakerRef{Data: wav, Filename: "reference.wav", ContentType: "audio/wav"}}
	data, contentType, err := speech.Multipart(r, wav, r.SpeakerRef.Filename, r.SpeakerRef.ContentType)
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct{ body *Body }{
		"success: recorded form":  {&Body{ContentType: "multipart/form-data; boundary=speech-reference", Data: recorded}},
		"success: generated form": {&Body{ContentType: contentType, Data: data}},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			req, err := c.NewRequest(t.Context(), http.MethodPost, "/tts", tc.body)
			if err != nil {
				t.Fatal(err)
			}
			defer req.Body.Close()
			original, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatal(err)
			}
			clone, err := requestForAttempt(req, 2)
			if err != nil {
				t.Fatal(err)
			}
			defer clone.Body.Close()
			data, err := io.ReadAll(clone.Body)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, original) {
				t.Fatal("the retried request differs from the original body")
			}
			if got, want := clone.Header.Get("Content-Type"), req.Header.Get("Content-Type"); got != want {
				t.Fatalf("retried Content-Type = %q, want %q", got, want)
			}
			_, params, err := mime.ParseMediaType(clone.Header.Get("Content-Type"))
			if err != nil {
				t.Fatal(err)
			}
			if boundary := params["boundary"]; boundary == "" || !bytes.Contains(data, []byte("--"+boundary+"\r\n")) {
				t.Fatalf("the retried body does not use the header boundary %q", boundary)
			}
			req.Body = io.NopCloser(bytes.NewReader(original))
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

func TestTTSSendRequest(t *testing.T) {
	ref := &SpeakerRef{Data: []byte("reference bytes"), Filename: "reference.wav", ContentType: "audio/wav"}
	tests := map[string]struct {
		req      *TTSRequest
		wantJSON bool
	}{
		"success: raw prompt with uploaded reference switches to JSON":        {req: &TTSRequest{Input: new("Hi"), RawPrompt: new(true), SpeakerRef: ref, SpeakerRefB64: []byte("stale")}, wantJSON: true},
		"success: explicit false raw prompt with uploaded reference too":      {req: &TTSRequest{Input: new("Hi"), RawPrompt: new(false), SpeakerRef: ref}, wantJSON: true},
		"success: uploaded reference without raw prompt stays multipart":      {req: &TTSRequest{Input: new("Hi"), SpeakerRef: ref}},
		"success: raw prompt without uploaded reference stays a JSON request": {req: &TTSRequest{Input: new("Hi"), RawPrompt: new(true)}},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			origB64 := string(tc.req.SpeakerRefB64)
			send, sendRef := ttsSendRequest(tc.req)
			if !tc.wantJSON {
				if send != tc.req || sendRef != tc.req.SpeakerRef {
					t.Fatal("the request or its reference was replaced without need")
				}
				return
			}
			if sendRef != nil {
				t.Fatal("a reference is still uploaded as multipart")
			}
			if send.SpeakerRef != nil || !bytes.Equal(send.SpeakerRefB64, ref.Data) {
				t.Fatalf("SpeakerRefB64 = %q, want the uploaded reference bytes", send.SpeakerRefB64)
			}
			if tc.req.SpeakerRef != ref || string(tc.req.SpeakerRefB64) != origB64 {
				t.Fatal("the caller's request was modified")
			}
			data, err := json.Marshal(send, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			var members map[string]jsontext.Value
			if err := json.Unmarshal(data, &members); err != nil {
				t.Fatal(err)
			}
			if want := strconv.FormatBool(*tc.req.RawPrompt); string(members["raw_prompt"]) != want {
				t.Fatalf("raw_prompt member = %s, want %s", members["raw_prompt"], want)
			}
			if string(members["speaker_ref_b64"]) != strconv.Quote(base64.StdEncoding.EncodeToString(ref.Data)) {
				t.Fatalf("speaker_ref_b64 member = %s, want the reference base64-encoded", members["speaker_ref_b64"])
			}
		})
	}
}

func ExampleClient_TTS() {
	c, err := New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	audio, err := c.TTS(context.Background(), &TTSRequest{Input: new("Hello")})
	if err != nil {
		log.Fatal(err)
	}
	if _, err := audio.WriteTo(os.Stdout); err != nil {
		log.Fatal(err)
	}
}
