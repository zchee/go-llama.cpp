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

// Package realtime is a client for the realtime WebSocket API of llama-server (/v1/realtime): live transcription
// and realtime conversation.
//
// Dial opens a Session. The context given to Dial governs only the handshake; the session stays open until the
// caller calls Close (a normal closure) or CloseNow.
package realtime

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"

	"github.com/coder/websocket"

	llama "github.com/zchee/go-llama.cpp"
)

// DefaultReadLimit is the largest message a Session accepts unless DialOptions.ReadLimit replaces it. Server
// events carry base64 audio, so the limit is far above the 32 KiB default of the WebSocket library.
const DefaultReadLimit = 32 << 20

// Intent values of DialOptions.
const (
	// IntentTranscription opens a transcription session.
	IntentTranscription = "transcription"
)

// DialOptions configures Dial.
type DialOptions struct {
	// Intent is the "intent" query parameter. IntentTranscription opens a transcription session; empty opens a
	// realtime conversation.
	Intent string
	// Model is the "model" query parameter, which a server in router mode uses to pick the model.
	Model string
	// ReadLimit is the largest message accepted, in bytes. Zero selects DefaultReadLimit.
	ReadLimit int64
	// RequestOptions are applied to the handshake request, for example llama.WithAutoload.
	RequestOptions []llama.RequestOption
}

// Session is an open realtime WebSocket connection.
type Session struct {
	conn *websocket.Conn
}

// Dial opens a realtime session through the HTTP client and the authentication headers of c. ctx bounds the
// handshake only. A handshake the server refuses with an error status is returned as an *llama.APIError.
//
// A Timeout of the HTTP client of c bounds the handshake only; the WebSocket library does not apply it to the
// session.
func Dial(ctx context.Context, c *llama.Client, opts *DialOptions) (*Session, error) {
	if opts == nil {
		opts = &DialOptions{}
	}
	reqOpts := opts.RequestOptions
	if opts.Intent != "" {
		reqOpts = append(reqOpts, llama.WithQuery("intent", opts.Intent))
	}
	if opts.Model != "" {
		reqOpts = append(reqOpts, llama.WithModelQuery(opts.Model))
	}
	req, err := c.NewRequest(ctx, http.MethodGet, "/v1/realtime", nil, reqOpts...)
	if err != nil {
		return nil, err
	}

	conn, resp, err := websocket.Dial(ctx, req.URL.String(), &websocket.DialOptions{
		HTTPClient: c.HTTPClient(),
		HTTPHeader: req.Header,
	})
	if err != nil {
		if resp != nil && resp.StatusCode >= http.StatusBadRequest {
			apiErr := llama.ReadAPIError(resp)
			return nil, fmt.Errorf("realtime: handshake refused: %w", apiErr)
		}
		return nil, fmt.Errorf("realtime: dial: %w", err)
	}

	limit := opts.ReadLimit
	if limit <= 0 {
		limit = DefaultReadLimit
	}
	conn.SetReadLimit(limit)
	return &Session{conn: conn}, nil
}

// Close ends the session with a normal closure and waits for the server to acknowledge it. When ctx ends first,
// the connection is closed without waiting.
func (s *Session) Close(ctx context.Context) error {
	done := make(chan error, 1)
	go func() { done <- s.conn.Close(websocket.StatusNormalClosure, "") }()
	select {
	case err := <-done:
		return normalizeCloseError(err)
	case <-ctx.Done():
		_ = s.conn.CloseNow()
		<-done
		return context.Cause(ctx)
	}
}

// CloseNow closes the connection without the closing handshake.
func (s *Session) CloseNow() error {
	return normalizeCloseError(s.conn.CloseNow())
}

// normalizeCloseError treats closing an already closed session as success.
func normalizeCloseError(err error) error {
	if err == nil || errors.Is(err, net.ErrClosed) {
		return nil
	}
	if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
		return nil
	}
	return err
}
