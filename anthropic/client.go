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

// Package anthropic is a client for the Anthropic-compatible endpoints of llama-server: messages and token
// counting.
//
// The server accepts the API key both as "Authorization: Bearer" and as "X-Api-Key"; build the llama.Client with
// llama.WithAPIKeyHeader(llama.APIKeyHeaderXAPIKey) to send the header Anthropic clients use. Authentication,
// retries and errors are those of the llama.Client the client is built from.
package anthropic

import (
	llama "github.com/zchee/go-llama.cpp"
)

// Client is a client for the Anthropic-compatible endpoints. It is safe for concurrent use.
type Client struct {
	c *llama.Client
}

// NewClient returns a client that sends requests through c.
func NewClient(c *llama.Client) *Client {
	return &Client{c: c}
}

// Llama returns the client the requests are sent through.
func (c *Client) Llama() *llama.Client {
	return c.c
}
