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

package harness

import (
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
)

// Observer is an http.RoundTripper that records every exchange and forwards it unchanged to the next
// RoundTripper. It never builds a response itself.
type Observer struct {
	next http.RoundTripper

	mu        sync.Mutex
	exchanges []*Exchange
}

// NewObserver returns an Observer that forwards to next.
func NewObserver(next http.RoundTripper) *Observer {
	return &Observer{next: next}
}

// Exchange is one observed request and its outcome.
type Exchange struct {
	// Method is the request method.
	Method string
	// Path is the request path.
	Path string
	// Query is the parsed request query.
	Query url.Values
	// Header is a copy of the request header.
	Header http.Header
	// StatusCode is the response status, zero when the round trip failed.
	StatusCode int
	// Err is the round trip error.
	Err error

	body *observedBody
}

// Upgrade reports whether the request asked for a WebSocket upgrade.
func (e *Exchange) Upgrade() bool {
	return strings.EqualFold(e.Header.Get("Upgrade"), "websocket")
}

// BodyClosed reports whether the response body has been closed. It is false when there was no response.
func (e *Exchange) BodyClosed() bool {
	return e.body != nil && e.body.closed.Load()
}

// BytesRead returns the number of response body bytes read so far.
func (e *Exchange) BytesRead() int64 {
	if e.body == nil {
		return 0
	}
	return e.body.read.Load()
}

// RoundTrip implements http.RoundTripper.
func (o *Observer) RoundTrip(req *http.Request) (*http.Response, error) {
	ex := &Exchange{
		Method: req.Method,
		Path:   req.URL.Path,
		Query:  req.URL.Query(),
		Header: req.Header.Clone(),
	}
	o.mu.Lock()
	o.exchanges = append(o.exchanges, ex)
	o.mu.Unlock()

	resp, err := o.next.RoundTrip(req)
	if err != nil {
		o.mu.Lock()
		ex.Err = err
		o.mu.Unlock()
		return nil, err
	}
	if resp.Body == nil {
		o.mu.Lock()
		ex.StatusCode = resp.StatusCode
		o.mu.Unlock()
		return resp, nil
	}
	b := &observedBody{rc: resp.Body}
	o.mu.Lock()
	ex.StatusCode = resp.StatusCode
	ex.body = b
	o.mu.Unlock()
	// A 101 Switching Protocols body is the hijacked connection, which WebSocket libraries write to; keep it
	// writable.
	if rwc, ok := resp.Body.(io.ReadWriteCloser); ok {
		resp.Body = &observedRWBody{observedBody: b, w: rwc}
	} else {
		resp.Body = b
	}
	return resp, nil
}

// Exchanges returns the exchanges observed so far, oldest first.
func (o *Observer) Exchanges() []*Exchange {
	o.mu.Lock()
	defer o.mu.Unlock()
	return slices.Clone(o.exchanges)
}

// Reset forgets the observed exchanges.
func (o *Observer) Reset() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.exchanges = nil
}

// observedBody records reads and Close of a response body.
type observedBody struct {
	rc     io.ReadCloser
	read   atomic.Int64
	closed atomic.Bool
}

func (b *observedBody) Read(p []byte) (int, error) {
	n, err := b.rc.Read(p)
	b.read.Add(int64(n))
	return n, err
}

func (b *observedBody) Close() error {
	b.closed.Store(true)
	return b.rc.Close()
}

// observedRWBody is an observedBody that stays writable.
type observedRWBody struct {
	*observedBody
	w io.Writer
}

func (b *observedRWBody) Write(p []byte) (int, error) { return b.w.Write(p) }
