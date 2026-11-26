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

package realtime

import (
	"context"
	"encoding/json/jsontext"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
)

func TestSendRejectsExtraCollisions(t *testing.T) {
	c, err := llama.New("http://localhost")
	if err != nil {
		t.Fatal(err)
	}
	// A rejected event must return before accessing the WebSocket connection.
	s := &Session{client: c}
	tests := map[string]struct {
		send func(context.Context) error
		want string
	}{
		"error: Send": {
			send: func(ctx context.Context) error {
				return s.Send(ctx, ClientEvent{Type: InputAudioBufferAppend, Extra: map[string]jsontext.Value{"audio": jsontext.Value(`"AAE="`)}})
			},
			want: `llama: realtime.ClientEvent.Extra contains "audio"; use field Audio instead`,
		},
		"error: UpdateSession": {
			send: func(ctx context.Context) error {
				return s.UpdateSession(ctx, SessionConfig{Extra: map[string]jsontext.Value{"max_output_tokens": jsontext.Value(`7`)}})
			},
			want: `llama: realtime.SessionConfig.Extra contains "max_output_tokens"; use field MaxOutputTokens instead`,
		},
		"error: UpdateSession voice reference": {
			send: func(ctx context.Context) error {
				voice := ReferencedVoice("demo")
				voice.Reference.Extra = map[string]jsontext.Value{"id": jsontext.Value(`"other"`)}
				return s.UpdateSession(ctx, SessionConfig{Audio: &AudioConfig{Output: &AudioOutput{Voice: Nullable[Voice]{Value: &voice}}}})
			},
			want: `llama: realtime.VoiceReference.Extra contains "id"; use field ID instead`,
		},
		"error: CreateItem": {
			send: func(ctx context.Context) error {
				return s.CreateItem(ctx, Item{Type: "message", Extra: map[string]jsontext.Value{"role": jsontext.Value(`"user"`)}})
			},
			want: `llama: realtime.Item.Extra contains "role"; use field Role instead`,
		},
		"error: CreateResponse": {
			send: func(ctx context.Context) error {
				return s.CreateResponse(ctx, &ResponseConfig{Extra: map[string]jsontext.Value{"tool_choice": jsontext.Value(`"auto"`)}})
			},
			want: `llama: realtime.ResponseConfig.Extra contains "tool_choice"; use field ToolChoice instead`,
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			if err := tc.send(t.Context()); err == nil || err.Error() != tc.want {
				t.Fatalf("send returned %v, want %q", err, tc.want)
			}
		})
	}
}
