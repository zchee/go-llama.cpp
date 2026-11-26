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
	"testing"

	gocmp "github.com/google/go-cmp/cmp"
)

func TestAudioFormat(t *testing.T) {
	tests := map[string]struct {
		header    string
		want      AudioFormat
		wantError bool
	}{
		"success: WAV metadata belongs to the header in the body": {header: "audio/wav", want: AudioFormat{MediaType: "audio/wav"}},
		"success: mono when channels omitted":                     {header: "audio/pcm;rate=24000;bits=16;encoding=signed-int", want: AudioFormat{MediaType: "audio/pcm", SampleRate: 24000, Bits: 16, Encoding: "signed-int", Channels: 1}},
		"success: stereo float samples":                           {header: `audio/pcm; channels=2; bits=32; encoding="float"; rate=44100`, want: AudioFormat{MediaType: "audio/pcm", SampleRate: 44100, Bits: 32, Encoding: "float", Channels: 2}},
		"error: no media type":                                    {wantError: true},
		"error: JSON is not audio":                                {header: "application/json", wantError: true},
		"error: missing sample rate":                              {header: "audio/pcm;bits=16;encoding=signed-int", wantError: true},
		"error: negative channels":                                {header: "audio/pcm;rate=24000;bits=16;encoding=signed-int;channels=-1", wantError: true},
		"error: invalid bit depth":                                {header: "audio/pcm;rate=24000;bits=no;encoding=float", wantError: true},
		"error: missing encoding":                                 {header: "audio/pcm;rate=24000;bits=32", wantError: true},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := audioFormat(tc.header)
			if (err != nil) != tc.wantError {
				t.Fatalf("parsing %q: %v", tc.header, err)
			}
			if diff := gocmp.Diff(tc.want, got); diff != "" {
				t.Fatal(diff)
			}
		})
	}
}
