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
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strconv"
)

// Audio is an audio response, whether the server streamed it or generated it in one piece.
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

func audioFormat(contentType string) (AudioFormat, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return AudioFormat{}, err
	}
	f := AudioFormat{MediaType: mediaType}
	switch mediaType {
	case "audio/wav":
		return f, nil
	case "audio/pcm":
		f.Encoding = params["encoding"]
		if f.Encoding == "" {
			return AudioFormat{}, errors.New("llama: missing PCM encoding")
		}
		f.Channels = 1
		for key, dest := range map[string]*int{"rate": &f.SampleRate, "channels": &f.Channels, "bits": &f.Bits} {
			value, ok := params[key]
			if !ok && key == "channels" {
				continue
			}
			n, err := strconv.Atoi(value)
			if err != nil || n <= 0 {
				return AudioFormat{}, fmt.Errorf("llama: invalid PCM %s %q", key, value)
			}
			*dest = n
		}
		return f, nil
	default:
		return AudioFormat{}, fmt.Errorf("llama: unexpected speech media type %q", mediaType)
	}
}

// WriteTo copies the audio to w and closes Body even when reading or writing fails.
func (a *Audio) WriteTo(w io.Writer) (int64, error) {
	n, err := io.Copy(w, a.Body)
	return n, errors.Join(err, a.Body.Close())
}

// AudioCall posts a JSON speech request or, when ref is set, a replayable multipart request.
// The caller owns the returned audio body.
func AudioCall(ctx context.Context, c *Client, path string, request any, ref *SpeakerRef, opts ...RequestOption) (*Audio, error) {
	if ref != nil {
		body, err := ref.ttsMultipart(request)
		if err != nil {
			return nil, c.RedactError(err)
		}
		request = body
	}
	res, err := CallRaw(ctx, c, http.MethodPost, path, request, opts...)
	if err != nil {
		return nil, err
	}
	contentType := res.Header.Get("Content-Type")
	format, err := audioFormat(contentType)
	if err != nil {
		return nil, c.RedactError(errors.Join(err, res.Body.Close()))
	}
	return &Audio{Body: res.Body, ContentType: contentType, Format: format}, nil
}
