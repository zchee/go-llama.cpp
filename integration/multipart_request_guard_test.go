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
	"encoding/json/jsontext"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

func TestMultipartEndpointExtra(t *testing.T) {
	srv := presetStories260K.server(t)
	type extra = map[string]jsontext.Value
	stream := extra{"stream": jsontext.Value(`true`)}
	format := extra{"response_format": jsontext.Value(`"pcm"`)}
	tts := func(ref *llama.SpeakerRef) func(context.Context, *llama.Client, extra) (bool, error) {
		return func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
			r, err := c.TTS(ctx, &llama.TTSRequest{Input: new("Once"), SpeakerRef: ref, Extra: e})
			if r != nil {
				r.Body.Close()
			}
			return r == nil, err
		}
	}
	speech := func(ref *llama.SpeakerRef) func(context.Context, *llama.Client, extra) (bool, error) {
		return func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
			r, err := openai.NewClient(c).Speech(ctx, &openai.SpeechRequest{Input: new("Once"), SpeakerRef: ref, Extra: e})
			if r != nil {
				r.Body.Close()
			}
			return r == nil, err
		}
	}
	tests := map[string]struct {
		call  func(context.Context, *llama.Client, extra) (bool, error)
		extra extra
		field string
	}{
		"error: native multipart speech": {tts(&llama.SpeakerRef{}), format, "ResponseFormat"},
		"error: openai multipart speech": {speech(&llama.SpeakerRef{}), stream, "Stream"},
		"error: multipart voice id in object Extra": {
			call: func(ctx context.Context, c *llama.Client, e extra) (bool, error) {
				r, err := openai.NewClient(c).Speech(ctx, &openai.SpeechRequest{
					Input: new("Once"), SpeakerRef: &llama.SpeakerRef{},
					Voice: openai.SpeechVoice{Object: &openai.SpeechVoiceID{Extra: e}},
				})
				if r != nil {
					r.Body.Close()
				}
				return r == nil, err
			},
			extra: extra{"id": jsontext.Value(`"demo"`)}, field: "ID",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, obs := srv.Client(t)
			nilResponse, err := tt.call(t.Context(), c, tt.extra)
			if !nilResponse || err == nil || !strings.Contains(err.Error(), "use field "+tt.field) {
				t.Errorf("nil response = %t, error = %v; want nil and error naming %s", nilResponse, err, tt.field)
			}
			if got := len(obs.Exchanges()); got != 0 {
				t.Errorf("sent %d requests; want none", got)
			}
		})
	}
}
