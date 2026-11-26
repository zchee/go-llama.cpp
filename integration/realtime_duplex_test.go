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
	"encoding/base64"
	"iter"
	"os"
	"path/filepath"
	"testing"

	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/realtime"
)

func TestRealtimeFullDuplex(t *testing.T) {
	model := os.Getenv("LLAMA_TEST_FULL_DUPLEX_MODEL")
	if model == "" {
		t.Log("set LLAMA_TEST_FULL_DUPLEX_MODEL to exercise full-duplex audio")
		return
	}
	args := []string{"--parallel", "1"}
	// Hibiki distributions with a separate audio projector need both GGUF files.
	if projector := os.Getenv("LLAMA_TEST_FULL_DUPLEX_MMPROJ"); projector != "" {
		path, err := filepath.Abs(projector)
		if err != nil {
			t.Fatal(err)
		}
		args = append(args, "--mmproj", path)
	}
	srv := harness.StartT(t, harness.Options{Model: model, Args: args})
	c, _ := srv.Client(t)
	s := realtimeDial(t, c, nil)
	next, stop := iter.Pull2(s.Events())
	defer stop()
	read := func() realtime.ServerEvent {
		t.Helper()
		ev, err, ok := next()
		if err != nil || !ok || ev.Type == realtime.ErrorEvent {
			t.Fatalf("reading duplex event: ok=%t, err=%v, event=%+v", ok, err, ev)
		}
		return ev
	}
	if ev := read(); ev.Type != realtime.SessionCreated {
		t.Fatalf("first event = %s, want session.created", ev.Type)
	}
	pcm := realtimePCM()
	if err := s.AppendAudio(t.Context(), pcm); err != nil {
		t.Fatal(err)
	}
	opened := read()
	if opened.Type != realtime.ResponseCreated || opened.Response == nil || opened.Response.ID == "" {
		t.Fatalf("first audio append did not open a response: %+v", opened)
	}
	// Send more input to the open response without a turn commit or response.create.
	if err := s.AppendAudio(t.Context(), pcm); err != nil {
		t.Fatal(err)
	}
	id := opened.Response.ID
	var canceled, audioDone bool
	for {
		ev := read()
		switch ev.Type {
		case realtime.ResponseCreated:
			t.Fatal("an append opened a second response before cancellation")
		case realtime.ResponseOutputAudioDelta:
			if audioDone || ev.ResponseID == nil || *ev.ResponseID != id || ev.Delta == nil {
				t.Fatalf("unexpected audio delta: %+v", ev)
			}
			b, err := base64.StdEncoding.DecodeString(*ev.Delta)
			if err != nil || len(b) == 0 || len(b)%2 != 0 {
				t.Fatalf("invalid PCM16 audio delta (%d bytes): %v", len(b), err)
			}
			if !canceled {
				if err := s.CancelResponse(t.Context()); err != nil {
					t.Fatal(err)
				}
				canceled = true
			}
		case realtime.ResponseOutputAudioDone:
			if !canceled || audioDone || ev.ResponseID == nil || *ev.ResponseID != id {
				t.Fatalf("unexpected output_audio.done: %+v", ev)
			}
			audioDone = true
		case realtime.ResponseDone:
			if !audioDone || ev.Response == nil || ev.Response.ID != id || ev.Response.Status != "cancelled" {
				t.Fatalf("response.done after audio done=%t: %+v", audioDone, ev.Response)
			}
			return
		}
	}
}
