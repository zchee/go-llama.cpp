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
	"slices"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

// Adding or removing a recording requires updating both its wire tests and the inventory.
func TestRecordedFixtureInventory(t *testing.T) {
	var listed, onDisk []string
	for line := range strings.Lines(string(jsontest.ReadFile(t, "testdata/fixture-inventory.txt"))) {
		if strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			t.Fatalf("invalid fixture inventory row: %q", line)
		}
		for suffix := range strings.SplitSeq(fields[1], ",") {
			for _, name := range fields[2:] {
				listed = append(listed, path.Join(fields[0], name+"."+suffix))
			}
		}
	}
	root := os.DirFS(".")
	for _, dir := range []string{"testdata", "anthropic/testdata", "openai/testdata", "realtime/testdata", "systemone/testdata"} {
		err := fs.WalkDir(root, dir, func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return err
			}
			switch path.Ext(file) {
			case ".json", ".sse":
				onDisk = append(onDisk, file)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	slices.Sort(listed)
	slices.Sort(onDisk)
	if diff := gocmp.Diff(listed, onDisk); diff != "" {
		t.Errorf("recording inventory (-listed +on disk):\n%s", diff)
	}
}

func TestRecordedRequestAndResponseObjects(t *testing.T) {
	tests := map[string]struct{ check func(*testing.T, []byte) }{
		"completion/children_stream.req.json":  {completionCheckWire[CompletionRequest]},
		"completion/projected_stream.req.json": {completionCheckWire[CompletionRequest]},
		"infill/unsupported.req.json":          {completionCheckWire[InfillRequest]},
		"slots/invalid_filename.req.json":      {completionCheckWire[SlotFileRequest]},
		"slots/missing.req.json":               {completionCheckWire[SlotFileRequest]},
	}
	for path, tt := range tests {
		t.Run(path, func(t *testing.T) { tt.check(t, jsontest.ReadFile(t, "testdata/"+path)) })
	}
}

func TestRecordedMultiPromptStreamFrames(t *testing.T) {
	tests := map[string]struct{ path string }{
		"success: sibling completions":   {"testdata/completion/children_stream.resp.sse"},
		"success: projected completions": {"testdata/completion/projected_stream.resp.sse"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			frames := 0
			for _, ev := range jsontest.EventData(t, jsontest.ReadFile(t, tt.path)) {
				if jsontext.Value(ev.Data).Kind() == 'n' {
					continue
				}
				completionCheckWire[CompletionResponse](t, ev.Data)
				frames++
			}
			if frames == 0 {
				t.Fatal("missing completion frames")
			}
		})
	}
}

func TestRecordedCompletionStreamErrorEnvelope(t *testing.T) {
	events := jsontest.EventData(t, jsontest.ReadFile(t, "openai/testdata/completions/stream_error.resp.sse"))
	if len(events) == 0 {
		t.Fatal("missing stream error")
	}
	completionCheckWire[errorEnvelope](t, events[len(events)-1].Data)
}

func TestRecordedInvalidRequestTypes(t *testing.T) {
	tests := map[string]struct{ request any }{
		"detokenize/error":        {new(DetokenizeRequest)},
		"tokenize/error":          {new(TokenizeRequest)},
		"lora_adapters/non_array": {new([]LoRAWeight)},
	}
	for path, tt := range tests {
		t.Run(path, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/"+path+".req.json")
			if err := json.Unmarshal(data, tt.request); err == nil {
				t.Errorf("accepted deliberately invalid %T recording: %s", tt.request, data)
			}
		})
	}
}
