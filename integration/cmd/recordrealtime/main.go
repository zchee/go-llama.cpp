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

// Command recordrealtime records the JSON frames of real /v1/realtime sessions, in each direction, into
// realtime/testdata/realtime. Run it from the repository root: go run ./integration/cmd/recordrealtime [session].
package main

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coder/websocket"

	"github.com/zchee/go-llama.cpp/integration/harness"
)

type step struct{ send, until string }

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

// run records every session, or only the one named by the first argument.
func run(args []string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	pcm := make([]byte, 32000)
	for i := range 16000 {
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(0.3*32767*math.Sin(2*math.Pi*440*float64(i)/16000))))
	}
	audio := `{"type":"input_audio_buffer.append","audio":"` + base64.StdEncoding.EncodeToString(pcm) + `"}`
	const item = `{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}`
	const tool = `{"type":"function","name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string","enum":["Paris"]}},"required":["city"]}}`
	const done = "conversation.item.input_audio_transcription.completed"
	live := []step{
		{`{"type":"response.create"}`, "error"},
		{`{"type":"transcription_session.update","session":{"audio":{"input":{"format":{"rate":16000}}}}}`, "session.updated"},
		{`{"type":"input_audio_buffer.clear"}`, "input_audio_buffer.cleared"},
		{audio, ""},
		{`{"type":"input_audio_buffer.commit"}`, done},
		{audio, ""},
		{`{"type":"input_audio_buffer.commit","final":true}`, done},
	}
	var names []string
	for _, tc := range []struct {
		name, model, query string
		steps              []step
	}{
		{"voxtral", "tinyVoxtral-Realtime", "", live},
		{"live-asr", "tinyLFM3-Omni", "?intent=transcription", live},
		{"conversation", "tinyLFM3-Omni", "", []step{
			{`{"type":"response.create"}`, "error"},
			{`{"type":"session.update","session":{"instructions":"Be brief.","audio":{"output":{"voice":{"id":"male"},"language":"en"}},"tools":[` + tool + `],"max_output_tokens":"inf","tool_choice":"none"}}`, "session.updated"},
			{`{"type":"session.update","session":{"audio":{"output":{"voice":"male"}},"output_modalities":["text"],"tools":[]}}`, "session.updated"},
			{item, "conversation.item.done"},
			{`{"type":"response.create","response":{"max_output_tokens":8}}`, "response.done"},
			{audio, ""},
			{`{"type":"response.create","response":{"output_modalities":["audio"],"max_output_tokens":8}}`, "response.done"},
			{item, "conversation.item.done"},
			{`{"type":"response.create"}`, "response.output_text.delta"},
			{`{"type":"response.cancel"}`, "response.done"},
		}},
		{"turn-asr", "tinyLFM3-Omni", "", []step{
			{`{"type":"session.update","session":{"type":"transcription","max_output_tokens":8}}`, "session.updated"},
			{`{"type":"session.update","session":{"audio":{"input":{"transcription":{"language":"fr","prompt":"Transcribe"}}}}}`, "session.updated"},
			{audio, ""},
			{`{"type":"input_audio_buffer.commit"}`, done},
		}},
		{"tools", "tinyLFM3-Omni", "", []step{
			{`{"type":"session.update","session":{"tools":[{"type":"function","name":"weather","defer_loading":false,"strict":true,"parameters":{"type":"object","properties":{},"additionalProperties":false}}],"tool_choice":"required","output_modalities":["text"],"max_output_tokens":16}}`, "session.updated"},
			{item, "conversation.item.done"},
			{`{"type":"response.create"}`, "response.done"},
			{"tool result", "conversation.item.done"},
		}},
	} {
		if len(args) > 1 || (len(args) == 1 && tc.name != args[0]) {
			names = append(names, tc.name)
			continue
		}
		if err := record(ctx, tc.name, tc.model, tc.query, tc.steps); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if len(args) == 1 {
			return nil
		}
	}
	if len(names) > 0 {
		return fmt.Errorf("usage: recordrealtime [%s]; got %q", strings.Join(names, "|"), args)
	}
	return nil
}

func record(ctx context.Context, name, model, query string, steps []step) error {
	modelPath, err := harness.ModelPath(ctx, model)
	if err != nil {
		return err
	}
	proj, err := harness.ModelPath(ctx, model+"-mmproj")
	if err != nil {
		return err
	}
	args := []string{"--mmproj", proj, "--alias", model, "-c", "1024", "-b", "512", "-np", "1", "--seed", "42", "--jinja", "--reasoning", "off"}
	// The tiny model otherwise repeats its end-of-thinking token instead of opening a tool call.
	// These token IDs belong to the pinned LFM3 vocabulary (<tool_call> and </think>).
	if name == "tools" {
		args = append(args, "--logit-bias", "125019+10", "--logit-bias", "124902-inf")
	}
	srv, err := harness.Start(ctx, harness.Options{Model: modelPath, Args: args})
	if err != nil {
		return err
	}
	defer func() {
		if err := srv.Stop(); err != nil {
			log.Printf("stopping the server: %v", err)
		}
	}()
	conn, _, err := websocket.Dial(ctx, srv.URL+"/v1/realtime"+query, nil)
	if err != nil {
		return err
	}
	// The session is over by the time this runs; a second close only reports that it is already closed.
	defer func() { _ = conn.CloseNow() }()
	conn.SetReadLimit(32 << 20)
	var sent, received []jsontext.Value
	var callID string
	for _, s := range append([]step{{"", "session.created"}}, steps...) {
		if s.send == "tool result" {
			if callID == "" {
				return fmt.Errorf("response finished without a tool call")
			}
			s.send = fmt.Sprintf(`{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":%q,"output":"sunny"}}`, callID)
		}
		if s.send != "" {
			b := []byte(s.send)
			if err := conn.Write(ctx, websocket.MessageText, b); err != nil {
				return err
			}
			sent = append(sent, b)
		}
		for s.until != "" {
			_, b, err := conn.Read(ctx)
			if err != nil {
				return fmt.Errorf("waiting for %s: %w", s.until, err)
			}
			received = append(received, b)
			var ev struct {
				Type   string `json:"type"`
				CallID string `json:"call_id"`
			}
			if err := json.Unmarshal(b, &ev); err != nil {
				return err
			}
			if ev.Type == "response.function_call_arguments.done" {
				callID = ev.CallID
			}
			if ev.Type == s.until {
				break
			}
			if ev.Type == "error" {
				return fmt.Errorf("waiting for %s: %s", s.until, b)
			}
		}
	}
	dir := "realtime/testdata/realtime"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for suffix, frames := range map[string][]jsontext.Value{"client": sent, "server": received} {
		b, err := json.Marshal(frames, jsontext.WithIndent("  "))
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, name+"."+suffix+".json"), append(b, '\n'), 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("%s: %d sent, %d received\n", name, len(sent), len(received))
	return nil
}
