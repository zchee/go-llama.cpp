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
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json/jsontext"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/realtime"
)

func realtimePreset(model string) *serverPreset {
	return &serverPreset{name: "realtime/" + model, options: func(t testing.TB) harness.Options {
		return harness.Options{Model: harness.Model(t, model), Args: []string{"--mmproj", harness.Model(t, model+"-mmproj"), "--alias", model, "-c", "1024", "-b", "512", "-np", "1", "--jinja", "--reasoning", "off", "--seed", "42"}}
	}}
}

var (
	presetRealtimeOmni    = realtimePreset("tinyLFM3-Omni")
	presetRealtimeVoxtral = realtimePreset("tinyVoxtral-Realtime")
)

func realtimeDial(t *testing.T, c *llama.Client, opts *realtime.DialOptions) *realtime.Session {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	s, err := realtime.Dial(ctx, c, opts)
	cancel()
	if err != nil {
		t.Fatal(err)
	}
	timer := time.AfterFunc(time.Minute, func() { _ = s.CloseNow() })
	t.Cleanup(func() {
		if !timer.Stop() {
			t.Error("realtime session watchdog fired before the test finished")
		}
		if err := s.CloseNow(); err != nil {
			t.Error(err)
		}
	})
	return s
}

func realtimeUntil(t *testing.T, s *realtime.Session, kind realtime.EventType) realtime.ServerEvent {
	t.Helper()
	for ev, err := range s.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == kind {
			return ev
		}
		if ev.Type == realtime.ErrorEvent {
			t.Fatalf("while waiting for %s: %+v", kind, ev.Error)
		}
	}
	t.Fatalf("connection closed before %s", kind)
	return realtime.ServerEvent{}
}

func realtimePCM() []byte {
	b := make([]byte, 32000)
	for i := range len(b) / 2 {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(0.3*32767*math.Sin(2*math.Pi*440*float64(i)/16000))))
	}
	return b
}

func TestRealtimeTranscription(t *testing.T) {
	tests := map[string]struct {
		preset *serverPreset
		intent string
	}{"success: streaming encoder": {presetRealtimeVoxtral, ""}, "success: live ASR intent": {presetRealtimeOmni, realtime.IntentTranscription}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := tt.preset.server(t).Client(t)
			s := realtimeDial(t, c, &realtime.DialOptions{Intent: tt.intent})
			created := realtimeUntil(t, s, realtime.SessionCreated)
			if created.Session == nil || created.Session.Object == nil || *created.Session.Object != "realtime.transcription_session" {
				t.Fatalf("unexpected session: %+v", created.Session)
			}
			if err := s.AppendAudio(t.Context(), realtimePCM()); err != nil {
				t.Fatal(err)
			}
			if err := s.Commit(t.Context(), true); err != nil {
				t.Fatal(err)
			}
			var deltas strings.Builder
			var completed bool
			for ev, err := range s.Events() {
				if err != nil {
					t.Fatal(err)
				}
				switch ev.Type {
				case realtime.ErrorEvent:
					t.Fatalf("transcription: %+v", ev.Error)
				case realtime.TranscriptionDelta:
					if ev.Delta == nil {
						t.Fatalf("transcription delta is missing text: %+v", ev)
					}
					deltas.WriteString(*ev.Delta)
				case realtime.TranscriptionCompleted:
					if ev.Transcript == nil || deltas.Len() == 0 || *ev.Transcript != deltas.String() {
						t.Fatalf("transcript does not match %q: %+v", deltas.String(), ev)
					}
					completed = true
				}
			}
			if !completed {
				t.Fatal("stream ended without completed transcript")
			}
		})
	}
}

func TestRealtimeConversation(t *testing.T) {
	tests := map[string]struct{ mode string }{"success: text response": {"text"}, "success: audio response": {"audio"}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := presetRealtimeOmni.server(t).Client(t)
			s := realtimeDial(t, c, nil)
			realtimeUntil(t, s, realtime.SessionCreated)
			if err := s.UpdateSession(t.Context(), realtime.SessionConfig{OutputModalities: realtime.Nullable[[]string]{Value: new([]string{tt.mode})}, MaxOutputTokens: new(realtime.MaxOutputTokens(8))}); err != nil {
				t.Fatal(err)
			}
			cfg := realtimeUntil(t, s, realtime.SessionUpdated).Session
			if cfg == nil || cfg.MaxOutputTokens == nil || *cfg.MaxOutputTokens != 8 {
				t.Fatalf("update not applied: %+v", cfg)
			}
			if err := s.CreateResponse(t.Context(), nil); err != nil {
				t.Fatal(err)
			}
			bad := realtimeUntil(t, s, realtime.ErrorEvent)
			if bad.Error == nil || bad.Error.Type != "conversation_empty" {
				t.Fatalf("unexpected error event: %+v", bad.Error)
			}
			if tt.mode == "audio" {
				if err := s.AppendAudio(t.Context(), realtimePCM()); err != nil {
					t.Fatal(err)
				}
			} else if err := s.CreateItem(t.Context(), realtime.Item{Type: "message", Role: new("user"), Content: []realtime.ContentPart{{Type: "input_text", Text: new("hello")}}}); err != nil {
				t.Fatal(err)
			}
			if err := s.CreateResponse(t.Context(), nil); err != nil {
				t.Fatal(err)
			}
			var bytes int
			for ev, err := range s.Events() {
				if err != nil {
					t.Fatal(err)
				}
				switch ev.Type {
				case realtime.ResponseOutputAudioDelta:
					if ev.Delta == nil {
						t.Fatalf("audio delta is missing audio: %+v", ev)
					}
					b, err := base64.StdEncoding.DecodeString(*ev.Delta)
					if err != nil || len(b)%2 != 0 || tt.mode != "audio" {
						t.Fatalf("bad audio delta (%d bytes): %v", len(b), err)
					}
					bytes += len(b)
				case realtime.ResponseOutputTextDelta:
					if tt.mode != "text" {
						t.Fatal("text in audio response")
					}
					if ev.Delta == nil {
						t.Fatalf("text delta is missing text: %+v", ev)
					}
					bytes += len(*ev.Delta)
				case realtime.ErrorEvent:
					t.Fatalf("response: %+v", ev.Error)
				case realtime.ResponseDone:
					if bytes == 0 || ev.Response == nil || ev.Response.Status != "completed" || len(ev.Response.Output) != 1 {
						t.Fatalf("unexpected response after %d bytes: %+v", bytes, ev.Response)
					}
					return
				}
			}
			t.Fatal("response.done missing")
		})
	}
}

func TestRealtimeOwnership(t *testing.T) {
	c, obs := presetRealtimeOmni.server(t).Client(t)
	s := realtimeDial(t, c, &realtime.DialOptions{Model: "tinyLFM3-Omni", RequestOptions: []llama.RequestOption{llama.WithHeader("X-Client-Test", "session")}})
	for ev, err := range s.Events() {
		if err != nil || ev.Type != realtime.SessionCreated {
			t.Fatalf("first event = %s, %v", ev.Type, err)
		}
		var rejected bool
		for _, err := range s.Events() {
			rejected = errors.Is(err, realtime.ErrConcurrentReader)
		}
		if !rejected {
			t.Fatal("overlapping reader not rejected")
		}
		break
	}
	const sends = 4
	sent := make(chan error, sends)
	for range sends {
		go func() { sent <- s.Send(t.Context(), realtime.ClientEvent{Type: realtime.InputAudioBufferClear}) }()
	}
	for range sends {
		realtimeUntil(t, s, realtime.InputAudioBufferCleared)
	}
	for range sends {
		if err := <-sent; err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	ex := obs.Exchanges()
	if len(ex) != 1 || ex[0].Method != "GET" || ex[0].Path != "/v1/realtime" || !ex[0].Upgrade() || !ex[0].BodyClosed() || ex[0].Header.Get("X-Client-Test") != "session" {
		t.Fatalf("unexpected upgrade or ownership: %+v", ex)
	}
	if got := ex[0].Query["model"]; !slices.Equal(got, []string{"tinyLFM3-Omni"}) {
		t.Fatalf("handshake model query = %q, want tinyLFM3-Omni", got)
	}
}

func TestRealtimeRejectsExtraBeforeWriting(t *testing.T) {
	s, body := observedRealtimeDial(t)
	realtimeUntil(t, s, realtime.SessionCreated)
	before := body.bytesWritten.Load()
	err := s.UpdateSession(t.Context(), realtime.SessionConfig{Extra: map[string]jsontext.Value{"max_output_tokens": jsontext.Value(`7`)}})
	if err == nil || err.Error() != `llama: realtime.SessionConfig.Extra contains "max_output_tokens"; use field MaxOutputTokens instead` {
		t.Errorf("UpdateSession = %v, want an Extra collision error", err)
	}
	if n := body.bytesWritten.Load() - before; n != 0 {
		t.Errorf("rejected update wrote %d bytes to the WebSocket", n)
	}

	if err := s.UpdateSession(t.Context(), realtime.SessionConfig{MaxOutputTokens: new(realtime.MaxOutputTokens(8))}); err != nil {
		t.Fatal(err)
	}
	cfg := realtimeUntil(t, s, realtime.SessionUpdated).Session
	if cfg == nil || cfg.MaxOutputTokens == nil || *cfg.MaxOutputTokens != 8 {
		t.Fatalf("first update received by the server should set max_output_tokens to 8: %+v", cfg)
	}
}

func TestRealtimeRawEventBypassesExtraCheck(t *testing.T) {
	c, _ := presetRealtimeOmni.server(t).Client(t)
	s := realtimeDial(t, c, nil)
	realtimeUntil(t, s, realtime.SessionCreated)
	if err := s.Send(t.Context(), realtime.ClientEvent{
		Raw:   jsontext.Value(`{"type":"future.client_event"}`),
		Extra: map[string]jsontext.Value{"type": jsontext.Value(`"response.cancel"`)},
	}); err != nil {
		t.Fatal(err)
	}
	ev := realtimeUntil(t, s, realtime.ErrorEvent)
	if ev.Error == nil || ev.Error.Message != "unknown event type: future.client_event" {
		t.Fatalf("server did not receive the raw event: %+v", ev.Error)
	}
}

func TestRealtimeCancelResponse(t *testing.T) {
	c, _ := presetRealtimeOmni.server(t).Client(t)
	s := realtimeDial(t, c, nil)
	realtimeUntil(t, s, realtime.SessionCreated)
	if err := s.CreateItem(t.Context(), realtime.Item{Type: "message", Role: new("user"), Content: []realtime.ContentPart{{Type: "input_text", Text: new("hello")}}}); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateResponse(t.Context(), &realtime.ResponseConfig{OutputModalities: realtime.Nullable[[]string]{Value: new([]string{"text"})}}); err != nil {
		t.Fatal(err)
	}
	realtimeUntil(t, s, realtime.ResponseOutputTextDelta)
	if err := s.CancelResponse(t.Context()); err != nil {
		t.Fatal(err)
	}
	done := realtimeUntil(t, s, realtime.ResponseDone)
	if done.Response == nil || done.Response.Status != "cancelled" {
		t.Fatalf("cancelled response: %+v", done.Response)
	}
}

func TestRealtimeRefusedHandshake(t *testing.T) {
	srv := presetRealtimeOmni.server(t)
	c, err := llama.New(srv.URL + "/not-a-route")
	if err != nil {
		t.Fatal(err)
	}
	defer c.HTTPClient().CloseIdleConnections()
	s, err := realtime.Dial(t.Context(), c, nil)
	if s != nil {
		_ = s.CloseNow()
		t.Fatal("upgraded at missing route")
	}
	api, ok := errors.AsType[*llama.APIError](err)
	if !ok || api.StatusCode != 404 {
		t.Fatalf("handshake returned %v, want HTTP 404 APIError", err)
	}
}

func TestRealtimeReadLimit(t *testing.T) {
	c, _ := presetRealtimeOmni.server(t).Client(t)
	s := realtimeDial(t, c, &realtime.DialOptions{ReadLimit: 32})
	for _, err := range s.Events() {
		if err == nil {
			t.Fatal("oversized session.created accepted")
		}
		return
	}
	t.Fatal("read limit did not report an error")
}

// The underlying exchange is real. Hold only the read of the close acknowledgement,
// so cancelling Close has to interrupt a handshake already in progress.
type realtimeCloseTransport struct {
	http.RoundTripper
	body *realtimeCloseBody
}

type realtimeCloseBody struct {
	io.ReadWriteCloser
	closing      func()
	blockWrite   bool
	bytesWritten atomic.Int64
	closeCount   atomic.Int32
	closed       chan struct{}
	close        func() error
	timedOut     bool
}

func (tr *realtimeCloseTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	resp, err := tr.RoundTripper.RoundTrip(r)
	if err != nil {
		return nil, err
	}
	if rwc, ok := resp.Body.(io.ReadWriteCloser); ok && resp.StatusCode == http.StatusSwitchingProtocols {
		b := &realtimeCloseBody{ReadWriteCloser: rwc, closed: make(chan struct{})}
		b.close = sync.OnceValue(func() error { close(b.closed); return rwc.Close() })
		resp.Body = b
		tr.body = b
	}
	return resp, nil
}

func (b *realtimeCloseBody) Read(p []byte) (int, error) {
	if b.closing != nil {
		b.closing()
		select {
		case <-b.closed:
		case <-time.After(2 * time.Second):
			b.timedOut = true
		}
	}
	return b.ReadWriteCloser.Read(p)
}

func (b *realtimeCloseBody) Write(p []byte) (int, error) {
	if b.blockWrite {
		<-b.closed
		return 0, net.ErrClosed
	}
	n, err := b.ReadWriteCloser.Write(p)
	b.bytesWritten.Add(int64(n))
	return n, err
}

func (b *realtimeCloseBody) Close() error {
	b.closeCount.Add(1)
	return b.close()
}

func observedRealtimeDial(t *testing.T) (*realtime.Session, *realtimeCloseBody) {
	t.Helper()
	srv := presetRealtimeOmni.server(t)
	base, _ := srv.Client(t)
	tr := &realtimeCloseTransport{RoundTripper: base.HTTPClient().Transport}
	c, err := llama.New(srv.URL, llama.WithHTTPClient(&http.Client{Transport: tr}))
	if err != nil {
		t.Fatal(err)
	}
	s := realtimeDial(t, c, nil)
	if tr.body == nil {
		t.Fatal("upgrade was not observed")
	}
	return s, tr.body
}

func TestRealtimeCloseCancellationInterruptsHandshake(t *testing.T) {
	s, body := observedRealtimeDial(t)
	realtimeUntil(t, s, realtime.SessionCreated)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	body.closing = cancel
	if err := s.Close(ctx); !errors.Is(err, context.Canceled) || body.timedOut {
		t.Fatalf("Close = %v, waited for unresponsive peer: %t", err, body.timedOut)
	}
	if n := body.closeCount.Load(); n != 1 {
		t.Fatalf("upgraded transport closed %d times, want once", n)
	}
}

func TestRealtimeCloseInterruptsConcurrentHandshake(t *testing.T) {
	tests := map[string]struct{ canceled bool }{"success: immediate": {}, "success: already canceled": {true}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			s, body := observedRealtimeDial(t)
			entered := make(chan struct{})
			body.closing = sync.OnceFunc(func() { close(entered) })
			first := make(chan error, 1)
			go func() { first <- s.Close(t.Context()) }()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("graceful close never reached the acknowledgment read")
			}
			second := make(chan error, 1)
			go func() {
				if !tt.canceled {
					second <- s.CloseNow()
					return
				}
				ctx, cancel := context.WithCancel(t.Context())
				cancel()
				second <- s.Close(ctx)
			}()
			select {
			case err := <-second:
				if (tt.canceled && !errors.Is(err, context.Canceled)) || (!tt.canceled && err != nil) {
					t.Fatalf("interrupting close: %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("second close did not interrupt the handshake")
			}
			select {
			case <-first:
			case <-time.After(time.Second):
				t.Fatal("graceful close still waiting after interruption")
			}
			if n := body.closeCount.Load(); n != 1 {
				t.Fatalf("upgraded transport closed %d times, want once", n)
			}
		})
	}
}

func TestRealtimeCloseOnce(t *testing.T) {
	tests := map[string]struct{ immediate bool }{"success: graceful then immediate": {}, "success: immediate then graceful": {true}}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			s, body := observedRealtimeDial(t)
			graceful := func() error { return s.Close(t.Context()) }
			first, second := graceful, s.CloseNow
			if tt.immediate {
				first, second = second, first
			}
			for _, end := range []func() error{first, second} {
				if err := end(); err != nil {
					t.Fatal(err)
				}
				if n := body.closeCount.Load(); n != 1 {
					t.Fatalf("upgraded transport closed %d times, want once", n)
				}
			}
		})
	}
}

func TestRealtimeEventsAfterWriteDeadlineClosesConnection(t *testing.T) {
	s, body := observedRealtimeDial(t)
	realtimeUntil(t, s, realtime.SessionCreated)
	if err := s.CreateItem(t.Context(), realtime.Item{Type: "message", Role: new("user"), Content: []realtime.ContentPart{{Type: "input_text", Text: new("hello")}}}); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateResponse(t.Context(), &realtime.ResponseConfig{OutputModalities: realtime.Nullable[[]string]{Value: new([]string{"text"})}}); err != nil {
		t.Fatal(err)
	}
	realtimeUntil(t, s, realtime.ResponseOutputTextDelta)
	body.blockWrite = true
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	if err := s.CancelResponse(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("write: %v, want deadline exceeded", err)
	}
	for _, err := range s.Events() {
		if !errors.Is(err, net.ErrClosed) {
			t.Fatalf("read after write deadline: %v, want closed connection", err)
		}
		return
	}
	t.Fatal("unfinished response ended without error")
}

func TestRealtimeCloseNowReleasesSlot(t *testing.T) {
	c, _ := presetRealtimeOmni.server(t).Client(t)
	first := realtimeDial(t, c, nil)
	realtimeUntil(t, first, realtime.SessionCreated)
	if err := first.CloseNow(); err != nil {
		t.Fatal(err)
	}
	second := realtimeDial(t, c, nil)
	realtimeUntil(t, second, realtime.SessionCreated)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := second.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close(canceled) = %v", err)
	}
}
