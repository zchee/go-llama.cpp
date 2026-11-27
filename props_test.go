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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestPropsResponseFixtures(t *testing.T) {
	tests := map[string]struct {
		path      string
		wantUI    string
		wantVoice []string
	}{
		"success: ui_settings array": {
			path:   "testdata/props/ui_array.resp.json",
			wantUI: `[]`,
		},
		"success: ui_settings null": {
			path:   "testdata/props/ui_null.resp.json",
			wantUI: `null`,
		},
		"success: ui_settings object": {
			path:   "testdata/props/ui_object.resp.json",
			wantUI: `{"theme":"dark"}`,
		},
		"success: tts voices": {
			path:      "testdata/props/voices.resp.json",
			wantUI:    `{}`,
			wantVoice: []string{"female", "male"},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, tt.path)
			got := jsontest.CheckWire[PropsResponse](t, data)
			jsontest.CheckFallbacks[PropsResponse](t)

			out, err := json.Marshal(got.UISettings, jsonOptions)
			if err != nil {
				t.Fatalf("encoding ui_settings: %v", err)
			}
			gotUI := jsontext.Value(out)
			if err := gotUI.Canonicalize(); err != nil {
				t.Fatalf("canonicalizing ui_settings: %v", err)
			}
			if string(gotUI) != tt.wantUI {
				t.Errorf("UISettings = %s, want %s", gotUI, tt.wantUI)
			}
			if tt.wantVoice != nil {
				if diff := gocmp.Diff(tt.wantVoice, got.Voices); diff != "" {
					t.Errorf("Voices (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestPropsUISettingsRawValues(t *testing.T) {
	base := jsontest.ReadFile(t, "testdata/props/router.resp.json")
	var template map[string]jsontext.Value
	if err := json.Unmarshal(base, &template, jsonOptions); err != nil {
		t.Fatal(err)
	}

	tests := map[string]struct {
		raw string
	}{
		"success: null":    {raw: `null`},
		"success: array":   {raw: `[1,"a",{"k":true}]`},
		"success: object":  {raw: `{"theme":"dark","nested":{"x":2}}`},
		"success: string":  {raw: `"custom"`},
		"success: number":  {raw: `42`},
		"success: boolean": {raw: `true`},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			template["ui_settings"] = jsontext.Value(tt.raw)
			doc, err := json.Marshal(template, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			got := jsontest.RoundTrip[PropsResponse](t, doc)
			out, err := json.Marshal(got.UISettings, jsonOptions)
			if err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tt.raw, string(out)); diff != "" {
				t.Errorf("UISettings (-want +got):\n%s", diff)
			}
		})
	}
}
