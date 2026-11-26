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
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

var _ = registerContracts(map[string]contractCase{
	"POST /tts": {server: presetSpeech, call: func(t *testing.T, c *llama.Client) error {
		a, err := c.TTS(t.Context(), &llama.TTSRequest{Input: new("Hi"), NPredict: new(1)})
		if err != nil {
			return err
		}
		return a.Body.Close()
	}},
	"POST /v1/audio/speech": {server: presetSpeech, call: func(t *testing.T, c *llama.Client) error {
		a, err := openai.NewClient(c).Speech(t.Context(), &openai.SpeechRequest{Input: new("Hi"), NPredict: new(1)})
		if err != nil {
			return err
		}
		return a.Body.Close()
	}},
})
