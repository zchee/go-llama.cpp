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
	"bytes"
	"context"
	"encoding/json/jsontext"
	"errors"
	"io"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
)

// streamChunk is the part of a native /completion stream event these tests read.
type streamChunk struct {
	Content string                    `json:"content"`
	Stop    bool                      `json:"stop"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// streamRequest asks for a long stream: the end-of-sequence token is ignored, so the server generates until
// n_predict.
var streamRequest = map[string]any{
	"prompt":     "Once upon a time",
	"n_predict":  256,
	"ignore_eos": true,
	"stream":     true,
}

// Observe the socket write rather than a buffered HTTP write: the designated body byte has
// reached the connection, but net/http cannot have finished a longer upload yet.
type cancelWriteConn struct {
	net.Conn
	cancel context.CancelFunc
	after  byte
}

func (c *cancelWriteConn) Write(p []byte) (int, error) {
	n, err := c.Conn.Write(p)
	if n > 0 && p[n-1] == c.after {
		c.cancel()
	}
	return n, err
}

type streamRequestBody struct {
	io.ReadCloser
	notify func()
}

func (b *streamRequestBody) Close() error {
	err := b.ReadCloser.Close()
	b.notify()
	return err
}

type streamRequestTransport struct {
	*http.Transport
	closed chan struct{}
}

func (tr *streamRequestTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Body = &streamRequestBody{req.Body, sync.OnceFunc(func() { close(tr.closed) })}
	return tr.Transport.RoundTrip(req)
}

func TestStreamCancelInFlight(t *testing.T) {
	srv := presetStories260K.server(t)
	tests := map[string]struct{ after byte }{
		"error: during upload":       {after: ' '},
		"error: waiting for headers": {after: '}'},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			tr := &streamRequestTransport{http.DefaultTransport.(*http.Transport).Clone(), make(chan struct{})}
			dial := tr.DialContext
			tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				conn, err := dial(ctx, network, addr)
				if err != nil {
					return nil, err
				}
				return &cancelWriteConn{conn, cancel, tt.after}, nil
			}
			t.Cleanup(tr.CloseIdleConnections)
			c, err := llama.New(srv.URL, llama.WithHTTPClient(&http.Client{Transport: tr, Timeout: 10 * time.Second}))
			if err != nil {
				t.Fatal(err)
			}
			// JSON whitespace larger than the transport buffer makes the upload span multiple socket writes.
			// A non-streamed completion holds the response headers until generation ends.
			body := llama.Body{ContentType: "application/json", Data: append(bytes.Repeat([]byte{' '}, 128<<10), `{"prompt":"Once upon a time","n_predict":256,"ignore_eos":true}`...)}
			var yields int
			for event, err := range llama.Stream[streamChunk](ctx, c, http.MethodPost, "/completion", body, llama.WithNoRetry()) {
				yields++
				if event != nil || !errors.Is(err, context.Canceled) {
					t.Errorf("stream yielded (%v, %v), want (nil, context.Canceled)", event, err)
				}
			}
			if yields != 1 {
				t.Errorf("stream yielded %d results, want one cancellation", yields)
			}
			select {
			case <-tr.closed:
			case <-time.After(time.Second):
				t.Error("request body was not closed")
			}
		})
	}
}

// TestStreamOwnership checks when a stream opens and closes its response against a real server, through the
// low-level bridge so that it does not depend on any typed method.
func TestStreamOwnership(t *testing.T) {
	srv := presetStories260K.server(t)

	t.Run("success: never ranged sends nothing", func(t *testing.T) {
		c, obs := srv.Client(t)
		seq := llama.Stream[streamChunk](t.Context(), c, http.MethodPost, "/completion", streamRequest)
		_ = seq
		if n := len(obs.Exchanges()); n != 0 {
			t.Errorf("a stream that was never ranged sent %d requests", n)
		}
	})

	t.Run("success: full range closes the body and a second range yields ErrStreamConsumed", func(t *testing.T) {
		c, obs := srv.Client(t)
		seq := llama.Stream[streamChunk](t.Context(), c, http.MethodPost, "/completion", streamRequest)
		events, sawStop := 0, false
		for ev, err := range seq {
			if err != nil {
				t.Fatalf("stream error after %d events: %v", events, err)
			}
			events++
			sawStop = sawStop || ev.Stop
		}
		if events < 2 || !sawStop {
			t.Errorf("got %d events, final stop event seen %t; want a complete stream", events, sawStop)
		}
		ex := obs.Exchanges()
		if len(ex) != 1 || !ex[0].BodyClosed() {
			t.Fatalf("after a full range: %d requests, body closed %t; want 1 and true", len(ex), len(ex) == 1 && ex[0].BodyClosed())
		}

		var errs []error
		for v, err := range seq {
			if v != nil {
				t.Errorf("the second range yielded an event")
			}
			errs = append(errs, err)
		}
		if len(errs) != 1 || !errors.Is(errs[0], llama.ErrStreamConsumed) {
			t.Errorf("second range yielded %v, want [ErrStreamConsumed]", errs)
		}
		if n := len(obs.Exchanges()); n != 1 {
			t.Errorf("the second range sent a request: %d requests", n)
		}
	})

	t.Run("success: break closes the body", func(t *testing.T) {
		c, obs := srv.Client(t)
		n := 0
		for _, err := range llama.Stream[streamChunk](t.Context(), c, http.MethodPost, "/completion", streamRequest) {
			if err != nil {
				t.Fatal(err)
			}
			n++
			if n == 2 {
				break
			}
		}
		ex := obs.Exchanges()
		if len(ex) != 1 || !ex[0].BodyClosed() {
			t.Fatalf("after break: %d requests, body closed %t; want 1 and true", len(ex), len(ex) == 1 && ex[0].BodyClosed())
		}
	})

	t.Run("error: context canceled while receiving closes the body", func(t *testing.T) {
		c, obs := srv.Client(t)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		n := 0
		var streamErr error
		for _, err := range llama.Stream[streamChunk](ctx, c, http.MethodPost, "/completion", streamRequest) {
			if err != nil {
				streamErr = err
				break
			}
			n++
			if n == 1 {
				cancel()
			}
		}
		// Events already buffered when the context ends may still be delivered, so the stream either ends with
		// the cancellation or completes; it never yields another kind of error.
		if streamErr != nil && !errors.Is(streamErr, context.Canceled) {
			t.Errorf("stream error = %v, want one matching context.Canceled", streamErr)
		}
		ex := obs.Exchanges()
		if len(ex) != 1 || !ex[0].BodyClosed() {
			t.Fatalf("after cancel: %d requests, body closed %t; want 1 and true", len(ex), len(ex) == 1 && ex[0].BodyClosed())
		}
	})

	t.Run("error: context canceled before the response", func(t *testing.T) {
		c, obs := srv.Client(t)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		var errs []error
		for v, err := range llama.Stream[streamChunk](ctx, c, http.MethodPost, "/completion", streamRequest) {
			if v != nil {
				t.Errorf("a canceled stream yielded an event")
			}
			errs = append(errs, err)
		}
		if len(errs) != 1 || !errors.Is(errs[0], context.Canceled) {
			t.Errorf("canceled stream yielded %v, want one error matching context.Canceled", errs)
		}
		for _, ex := range obs.Exchanges() {
			if ex.StatusCode != 0 {
				t.Errorf("a request with a canceled context got a response: %d", ex.StatusCode)
			}
		}
	})
}
