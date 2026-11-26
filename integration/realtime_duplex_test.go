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
	"errors"
	"fmt"
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
	appendAudio := func() error { return s.AppendAudio(t.Context(), pcm) }
	cancelResponse := func() error { return s.CancelResponse(t.Context()) }
	if err := checkDuplexResponse(opened.Response.ID, read, appendAudio, cancelResponse); err != nil {
		t.Fatal(err)
	}
}

func checkDuplexResponse(id string, read func() realtime.ServerEvent, appendAudio, cancelResponse func() error) error {
	var appendedAgain, canceled, audioDone bool
	for {
		ev := read()
		switch ev.Type {
		case realtime.ResponseCreated:
			return errors.New("an append opened a second response before cancellation")
		case realtime.ResponseOutputAudioDelta:
			if audioDone || ev.ResponseID == nil || *ev.ResponseID != id || ev.Delta == nil {
				return fmt.Errorf("unexpected audio delta: %+v", ev)
			}
			b, err := base64.StdEncoding.DecodeString(*ev.Delta)
			if err != nil {
				return fmt.Errorf("decoding audio delta: %w", err)
			}
			if len(b) == 0 || len(b)%2 != 0 {
				return fmt.Errorf("invalid PCM16 audio delta length: %d", len(b))
			}
			if !appendedAgain {
				if err := appendAudio(); err != nil {
					return err
				}
				appendedAgain = true
				continue
			}
			if !canceled {
				if err := cancelResponse(); err != nil {
					return err
				}
				canceled = true
			}
		case realtime.ResponseOutputAudioDone:
			if !canceled || audioDone || ev.ResponseID == nil || *ev.ResponseID != id {
				return fmt.Errorf("unexpected output_audio.done: %+v", ev)
			}
			audioDone = true
		case realtime.ResponseDone:
			if !audioDone || ev.Response == nil || ev.Response.ID != id || ev.Response.Status != "cancelled" {
				return fmt.Errorf("response.done after audio done=%t: %+v", audioDone, ev.Response)
			}
			return nil
		}
	}
}

func TestDuplexResponseSequence(t *testing.T) {
	tests := map[string]struct {
		deltas       int
		wantCancelAt int
	}{
		"error: output only before second append":        {deltas: 1},
		"success: output before and after second append": {deltas: 2, wantCancelAt: 2},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			const id = "response"
			var events []realtime.ServerEvent
			for range tt.deltas {
				events = append(events, realtime.ServerEvent{Type: realtime.ResponseOutputAudioDelta, ResponseID: new(id), Delta: new("AAA=")})
			}
			events = append(events, realtime.ServerEvent{Type: realtime.ResponseOutputAudioDone, ResponseID: new(id)},
				realtime.ServerEvent{Type: realtime.ResponseDone, Response: &realtime.Response{ID: id, Status: "cancelled"}})
			var reads, appendAt, cancelAt int
			read := func() realtime.ServerEvent {
				reads++
				return events[reads-1]
			}
			err := checkDuplexResponse(id, read, func() error { appendAt = reads; return nil }, func() error { cancelAt = reads; return nil })
			if (err != nil) != (tt.wantCancelAt == 0) || appendAt != 1 || cancelAt != tt.wantCancelAt {
				t.Errorf("second append after event %d, cancel after event %d, error %v", appendAt, cancelAt, err)
			}
		})
	}
}
