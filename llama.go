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

// Package llama is a client for the HTTP server of llama.cpp (llama-server).
//
// A Client talks to one server. Its methods cover the native endpoints (completion, tokenization, embeddings,
// reranking, speech, slots, metrics, LoRA adapters and the router's model management). The OpenAI-, Anthropic-
// and TypeSafe-compatible endpoints and the realtime WebSocket API live in the openai, anthropic, systemone and
// realtime sub-packages, whose clients are built from a Client:
//
//	c, err := llama.New("http://127.0.0.1:8080", llama.WithAPIKey(key))
//	if err != nil {
//		return err
//	}
//	if err := c.WaitReady(ctx); err != nil {
//		return err
//	}
//
// # Wire types
//
// Request and response types follow the JSON the server reads and writes. Every struct that maps a JSON object
// has an Extra field tagged `json:",embed"` that collects members the struct does not name, so that members added
// by newer servers survive a decode and encode round trip at every level. Before sending a typed JSON request,
// the client rejects an Extra key whose name exactly matches a JSON member of the same struct, including
// promoted members, even when the typed field is unset. Use the named Go field instead.
//
// Optional scalars are pointers; set them with new, for example Temperature: new(0.0). Values that take several
// JSON shapes, such as Prompt, LogitBias and StringOrSlice, are built with constructor functions.
//
// # Errors
//
// An error response, and an error event inside a stream, is returned as an *APIError, whose Type is one of the
// ErrorType constants:
//
//	if apiErr, ok := errors.AsType[*llama.APIError](err); ok && apiErr.Type == llama.ErrorTypeExceedContextSize {
//		// shorten the prompt
//	}
//
// # Streams
//
// Streaming methods return an iter.Seq2 that sends its request when the range loop starts and closes the
// response body when the loop ends for any reason, including break and context cancellation. A stream can be
// ranged over only once; a second range yields ErrStreamConsumed.
//
// # Retries
//
// Requests are retried with exponential backoff when the server answers 503 unavailable_error and, for
// idempotent methods, when the connection fails before a response arrives. Request bodies are kept in memory, so
// a retry resends the same bytes. Streams are not retried once their response has been accepted. See RetryPolicy,
// WithRetry and WithNoRetry.
//
// # Other routes
//
// Call, Stream and CallRaw reach any route with the same authentication, retry and error handling, including the
// aliases the server registers for documented endpoints (for example /v1/health) and routes without a typed
// method.
package llama

//go:generate go run ./internal/cmd/genendpoints -o testdata/endpoints.v1.1.0.txt
