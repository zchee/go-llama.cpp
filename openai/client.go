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

// Package openai is a client for the OpenAI-compatible endpoints of llama-server: models, completions, chat
// completions and their real-time control, responses, embeddings, speech, and token counting.
//
// The request types embed llama.SamplingParams, so the llama.cpp sampling extensions are set in the same literal
// as the OpenAI fields:
//
//	oc := openai.NewClient(c)
//	res, err := oc.ChatCompletions(ctx, &openai.ChatCompletionRequest{
//		Model:          "m",
//		Messages:       []openai.Message{{Role: "user", Content: openai.TextContent("Hello")}},
//		SamplingParams: llama.SamplingParams{Mirostat: new(2)},
//	})
//
// Authentication, retries and errors are those of the llama.Client the client is built from.
package openai

import (
	llama "github.com/zchee/go-llama.cpp"
)

// Client is a client for the OpenAI-compatible endpoints. It is safe for concurrent use.
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
