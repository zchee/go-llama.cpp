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
	"io/fs"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

// Fixture names describe scenarios, not HTTP statuses. Inspect the bodies so errors like
// unsupported.resp.json are checked against the same private wire type as error.resp.json.
func TestRecordedErrorEnvelopes(t *testing.T) {
	jsontest.CheckFallbacks[errorEnvelope](t)
	root := os.DirFS(".")
	var checked int
	err := fs.WalkDir(root, ".", func(dir string, entry fs.DirEntry, err error) error {
		if err != nil || !entry.IsDir() {
			return err
		}
		if dir != "." && strings.HasPrefix(entry.Name(), ".") {
			return fs.SkipDir
		}
		if !strings.Contains("/"+dir+"/", "/testdata/") {
			return nil
		}
		files, err := fs.Glob(root, path.Join(dir, "*.resp.json"))
		if err != nil {
			return err
		}
		for _, file := range files {
			body := jsontest.ReadFile(t, file)
			if jsontext.Value(body).Kind() != '{' {
				continue
			}
			var probe struct {
				Error jsontext.Value `json:"error"`
			}
			if err := json.Unmarshal(body, &probe); err != nil {
				t.Fatalf("%s: %v", file, err)
			}
			if len(probe.Error) == 0 || probe.Error.Kind() == 'n' {
				continue
			}
			checked++
			t.Run(file, func(t *testing.T) {
				jsontest.RoundTrip[errorEnvelope](t, body)
				jsontest.InjectUnknown[errorEnvelope](t, body)
				jsontest.NoUnknown[errorEnvelope](t, body, nil)
			})
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if checked == 0 {
		t.Fatal("no recorded error envelopes found in testdata")
	}
}
