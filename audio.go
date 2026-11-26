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

package llama

import (
	"errors"
	"io"
)

// Audio is a buffered or streamed audio response. If generation fails after streaming begins, the server
// ends Body with a clean EOF; the client cannot distinguish that failure from a complete response.
type Audio struct {
	// Body is owned by the caller and must be closed.
	Body io.ReadCloser
	// ContentType is the unmodified Content-Type header.
	ContentType string
	// Format describes the encoding advertised in ContentType.
	Format AudioFormat
}

// AudioFormat describes an audio response. For WAV the sample metadata is in the body instead.
type AudioFormat struct {
	// MediaType is audio/wav or audio/pcm.
	MediaType string
	// SampleRate is the number of PCM sample frames per second.
	SampleRate int
	// Channels is the number of interleaved PCM channels, one when the header omits it.
	Channels int
	// Encoding is the PCM representation, such as float or signed-int.
	Encoding string
	// Bits is the number of bits per PCM sample.
	Bits int
}

// WriteTo copies the audio to w and always closes Body. On a write error it stops without draining Body
// and returns the number of bytes written and the error. Any close error is joined with the copy error.
func (a *Audio) WriteTo(w io.Writer) (int64, error) {
	n, err := io.Copy(w, a.Body)
	return n, errors.Join(err, a.Body.Close())
}
