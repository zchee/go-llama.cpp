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
	json "encoding/json/v2"
	"errors"
	"fmt"
	"iter"

	"github.com/coder/websocket"
)

// ErrConcurrentReader is returned when another Events iteration is already reading the session.
var ErrConcurrentReader = errors.New("realtime: another Events iterator is active")

// Send writes a JSON text frame. Writes are serialized by the WebSocket connection and may run concurrently with Events.
func (s *Session) Send(ctx context.Context, ev ClientEvent) error {
	b, err := json.Marshal(ev)
	if err != nil {
		return s.client.RedactError(fmt.Errorf("realtime: encoding client event: %w", err))
	}
	return s.client.RedactError(s.conn.Write(ctx, websocket.MessageText, b))
}

// Events reads on the caller's goroutine until a normal close, an I/O or decoding error, or a break.
// A break leaves the next frame for a later iteration. Only one iteration may be active at a time.
// ErrorEvent is delivered as data because the server can accept further requests after an invalid event.
func (s *Session) Events() iter.Seq2[ServerEvent, error] {
	return func(yield func(ServerEvent, error) bool) {
		if !s.reading.CompareAndSwap(false, true) {
			yield(ServerEvent{}, ErrConcurrentReader)
			return
		}
		defer s.reading.Store(false)
		for {
			typ, b, err := s.conn.Read(context.Background())
			if err != nil {
				if !s.localClose.Load() && websocket.CloseStatus(err) != websocket.StatusNormalClosure {
					yield(ServerEvent{}, s.client.RedactError(fmt.Errorf("realtime: reading server event: %w", err)))
				}
				return
			}
			if typ != websocket.MessageText {
				yield(ServerEvent{}, fmt.Errorf("realtime: expected a text frame, got %s", typ))
				return
			}
			var ev ServerEvent
			if err := json.Unmarshal(b, &ev); err != nil {
				yield(ServerEvent{}, s.client.RedactError(fmt.Errorf("realtime: decoding server event: %w", err)))
				return
			}
			if !yield(ev, nil) {
				return
			}
		}
	}
}

// AppendAudio sends a chunk of little-endian, mono PCM16 audio at the negotiated input rate.
// In full duplex mode the first chunk opens the response automatically.
func (s *Session) AppendAudio(ctx context.Context, pcm []byte) error {
	return s.Send(ctx, ClientEvent{Type: InputAudioBufferAppend, Audio: pcm})
}

// Commit flushes pending audio. In live transcription, final ends the input; full duplex has no commits.
func (s *Session) Commit(ctx context.Context, final bool) error {
	return s.Send(ctx, ClientEvent{Type: InputAudioBufferCommit, Final: new(final)})
}

// UpdateSession applies a configuration change to the current session.
func (s *Session) UpdateSession(ctx context.Context, config SessionConfig) error {
	return s.Send(ctx, ClientEvent{Type: SessionUpdate, Session: &config})
}

// CreateItem appends a user message or function call output to the conversation.
func (s *Session) CreateItem(ctx context.Context, item Item) error {
	return s.Send(ctx, ClientEvent{Type: ConversationItemCreate, Item: &item})
}

// CreateResponse asks the model to answer the current turn. A nil config uses session defaults.
func (s *Session) CreateResponse(ctx context.Context, config *ResponseConfig) error {
	return s.Send(ctx, ClientEvent{Type: ResponseCreate, Response: config})
}

// CancelResponse stops the current response without closing the session.
func (s *Session) CancelResponse(ctx context.Context) error {
	return s.Send(ctx, ClientEvent{Type: ResponseCancel})
}
