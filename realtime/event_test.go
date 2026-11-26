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

package realtime_test

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"log"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/internal/jsontest"
	"github.com/zchee/go-llama.cpp/realtime"
)

func realtimeFrames(t testing.TB, name, direction string) []jsontext.Value {
	t.Helper()
	data := jsontest.ReadFile(t, "testdata/realtime/"+name+"."+direction+".json")
	var frames []jsontext.Value
	if err := json.Unmarshal(data, &frames); err != nil {
		t.Fatal(err)
	}
	if len(frames) == 0 {
		t.Fatal("empty recording:", name, direction)
	}
	return frames
}

func TestRealtimeWire(t *testing.T) {
	tests := map[string]struct{ name string }{"success: per-turn transcription": {"turn-asr"}, "success: live ASR": {"live-asr"}, "success: Voxtral transcription": {"voxtral"}, "success: conversation": {"conversation"}, "success: tools": {"tools"}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			for _, b := range realtimeFrames(t, tt.name, "client") {
				jsontest.RoundTrip[realtime.ClientEvent](t, b)
				jsontest.InjectUnknown[realtime.ClientEvent](t, b)
				jsontest.NoUnknown[realtime.ClientEvent](t, b, nil)
			}
			for _, b := range realtimeFrames(t, tt.name, "server") {
				jsontest.RoundTrip[realtime.ServerEvent](t, b)
				jsontest.InjectUnknown[realtime.ServerEvent](t, b)
				jsontest.NoUnknown[realtime.ServerEvent](t, b, nil)
			}
		})
	}
	jsontest.CheckFallbacks[realtime.ClientEvent](t)
	jsontest.CheckFallbacks[realtime.ServerEvent](t)
	jsontest.CheckFallbacks[realtime.VoiceReference](t)
}

// Duplex uses the same append, cancel, text and audio frames as a conversation but has no commit or create.
func TestRealtimeDuplexFrames(t *testing.T) {
	want := map[realtime.EventType]bool{realtime.InputAudioBufferAppend: true, realtime.ResponseCancel: true, realtime.ResponseCreated: true, realtime.ResponseOutputTextDelta: true, realtime.ResponseOutputAudioDelta: true, realtime.ResponseDone: true}
	got := map[realtime.EventType]bool{}
	for _, dir := range []string{"client", "server"} {
		for _, b := range realtimeFrames(t, "conversation", dir) {
			var kind struct {
				Type realtime.EventType `json:"type"`
			}
			if err := json.Unmarshal(b, &kind); err != nil {
				t.Fatal(err)
			}
			if want[kind.Type] {
				got[kind.Type] = true
			}
		}
	}
	if diff := gocmp.Diff(want, got); diff != "" {
		t.Fatal(diff)
	}
}

func realtimeFuzzValue[T any](t *testing.T, b []byte) {
	t.Helper()
	var first, second T
	if err := json.Unmarshal(b, &first); err != nil {
		return
	}
	enc, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(enc, &second); err != nil {
		t.Fatal(err)
	}
	if diff := gocmp.Diff(first, second, gocmp.Comparer(func(x, y jsontext.Value) bool {
		return x.Compact() == nil && y.Compact() == nil && string(x) == string(y)
	})); diff != "" {
		t.Fatal(diff)
	}
}

func TestRealtimeMaxOutputTokens(t *testing.T) {
	tests := map[string]struct {
		wire string
		want realtime.MaxOutputTokens
		bad  bool
	}{"success: one": {wire: `1`, want: 1}, "success: unlimited": {wire: `"inf"`, want: realtime.UnlimitedTokens}, "error: zero": {wire: `0`, bad: true}, "error: numeric minus one": {wire: `-1`, bad: true}, "error: fraction": {wire: `1.5`, bad: true}, "error: string number": {wire: `"8"`, bad: true}, "error: null": {wire: `null`, bad: true}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var v realtime.MaxOutputTokens
			err := json.Unmarshal([]byte(tt.wire), &v)
			if (err != nil) != tt.bad || (!tt.bad && v != tt.want) {
				t.Fatalf("decode %s = %v, %v", tt.wire, v, err)
			}
		})
	}
}

func TestRealtimeVoice(t *testing.T) {
	tests := map[string]struct {
		v    realtime.Voice
		want string
		bad  bool
	}{"success: name": {realtime.NamedVoice("male"), `"male"`, false}, "success: reference": {realtime.ReferencedVoice("male"), `{"id":"male"}`, false}, "error: unspecified": {bad: true}, "error: ambiguous": {v: realtime.Voice{Name: new("a"), Reference: &realtime.VoiceReference{ID: "b"}}, bad: true}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			b, err := json.Marshal(tt.v)
			if (err != nil) != tt.bad || (!tt.bad && string(b) != tt.want) {
				t.Fatalf("encode voice = %s, %v", b, err)
			}
		})
	}
}

func TestRealtimeToolFrames(t *testing.T) {
	var args strings.Builder
	var finished *realtime.ServerEvent
	var result *realtime.Item
	for _, b := range realtimeFrames(t, "tools", "server") {
		var ev realtime.ServerEvent
		if err := json.Unmarshal(b, &ev); err != nil {
			t.Fatal(err)
		}
		switch ev.Type {
		case realtime.ResponseFunctionCallArgumentsDelta:
			args.WriteString(*ev.Delta)
		case realtime.ResponseFunctionCallArgumentsDone:
			finished = &ev
		case realtime.ConversationItemDone:
			if ev.Item.Type == "function_call_output" {
				result = ev.Item
			}
		}
	}
	if args.Len() == 0 || finished == nil || finished.Arguments == nil || *finished.Arguments != args.String() || finished.CallID == nil || result == nil || result.CallID == nil || *result.CallID != *finished.CallID {
		t.Fatalf("tool call arguments or output linkage missing: %v, %v", finished, result)
	}
}

func FuzzRealtimeMaxOutputTokens(f *testing.F) {
	for _, b := range realtimeFrames(f, "conversation", "server") {
		var ev struct {
			Session struct {
				Limit jsontext.Value `json:"max_output_tokens"`
			} `json:"session"`
		}
		if err := json.Unmarshal(b, &ev); err != nil {
			f.Fatal(err)
		}
		if ev.Session.Limit != nil {
			f.Add([]byte(ev.Session.Limit))
		}
	}
	for _, b := range realtimeFrames(f, "conversation", "client") {
		var ev struct {
			Response struct {
				Limit jsontext.Value `json:"max_output_tokens"`
			} `json:"response"`
		}
		if err := json.Unmarshal(b, &ev); err != nil {
			f.Fatal(err)
		}
		if ev.Response.Limit != nil {
			f.Add([]byte(ev.Response.Limit))
		}
	}
	f.Fuzz(realtimeFuzzValue[realtime.MaxOutputTokens])
}

func FuzzRealtimeVoice(f *testing.F) {
	for _, b := range realtimeFrames(f, "conversation", "client") {
		var ev struct {
			Session struct {
				Audio struct {
					Output struct {
						Voice jsontext.Value `json:"voice"`
					} `json:"output"`
				} `json:"audio"`
			} `json:"session"`
		}
		if err := json.Unmarshal(b, &ev); err != nil {
			f.Fatal(err)
		}
		if ev.Session.Audio.Output.Voice != nil {
			f.Add([]byte(ev.Session.Audio.Output.Voice))
		}
	}
	f.Fuzz(realtimeFuzzValue[realtime.Voice])
}

func FuzzRealtimeNullable(f *testing.F) {
	for _, b := range realtimeFrames(f, "conversation", "server") {
		var ev struct {
			Previous jsontext.Value `json:"previous_item_id"`
		}
		if err := json.Unmarshal(b, &ev); err != nil {
			f.Fatal(err)
		}
		if ev.Previous != nil {
			f.Add([]byte(ev.Previous))
		}
	}
	f.Fuzz(realtimeFuzzValue[realtime.Nullable[string]])
}

func ExampleSession() {
	c, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		log.Print(err)
		return
	}
	ctx := context.Background()
	s, err := realtime.Dial(ctx, c, nil)
	if err != nil {
		log.Print(err)
		return
	}
	defer func() {
		if err := s.CloseNow(); err != nil {
			log.Print(err)
		}
	}()
	if err := s.CreateItem(ctx, realtime.Item{Type: "message", Role: new("user"), Content: []realtime.ContentPart{{Type: "input_text", Text: new("Hello")}}}); err != nil {
		log.Print(err)
		return
	}
	if err := s.CreateResponse(ctx, &realtime.ResponseConfig{OutputModalities: realtime.Nullable[[]string]{Value: new([]string{"text"})}}); err != nil {
		log.Print(err)
		return
	}
	for ev, err := range s.Events() {
		if err != nil {
			log.Print(err)
			return
		}
		switch ev.Type {
		case realtime.ResponseOutputTextDelta:
			fmt.Print(*ev.Delta)
		case realtime.ErrorEvent:
			log.Print(ev.Error.Message)
			return
		case realtime.ResponseDone:
			return
		}
	}
}
