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
	"encoding/json/jsontext"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/realtime"
)

// TestObserver drives a real server through an observed client and checks what the Observer records, including
// a WebSocket upgrade, whose connection must stay writable.
func TestObserver(t *testing.T) {
	srv := harness.StartT(t, harness.Options{Model: harness.Model(t, "stories260K"), Args: []string{"--ctx-size", "512", "--parallel", "1"}})
	c, obs := srv.Client(t, llama.WithUserAgent("observer-test"))

	if _, err := c.Health(t.Context(), llama.WithQuery("probe", "1")); err != nil {
		t.Fatal(err)
	}
	ex := obs.Exchanges()
	if len(ex) != 1 {
		t.Fatalf("observed %d exchanges, want 1", len(ex))
	}
	if ex[0].Method != http.MethodGet || ex[0].Path != "/health" || ex[0].Query.Get("probe") != "1" ||
		ex[0].Header.Get("User-Agent") != "observer-test" || ex[0].StatusCode != http.StatusOK {
		t.Errorf("exchange = %+v", ex[0])
	}
	if !ex[0].BodyClosed() || ex[0].BytesRead() == 0 {
		t.Errorf("Call must read and close the body: closed %t, read %d", ex[0].BodyClosed(), ex[0].BytesRead())
	}

	obs.Reset()
	s, err := realtime.Dial(t.Context(), c, nil)
	if err != nil {
		t.Fatalf("realtime.Dial through the observer: %v", err)
	}
	ex = obs.Exchanges()
	if len(ex) != 1 || !ex[0].Upgrade() || ex[0].StatusCode != http.StatusSwitchingProtocols {
		t.Errorf("handshake exchange = %+v, want one upgrade answered 101", ex)
	}
	if err := s.Close(t.Context()); err != nil {
		t.Errorf("Session.Close: %v", err)
	}
	if !ex[0].BodyClosed() {
		t.Errorf("closing the session did not close the upgraded connection")
	}
}

func TestStartFailsFast(t *testing.T) {
	_, err := harness.Start(t.Context(), harness.Options{Model: filepath.Join(t.TempDir(), "missing.gguf")})
	if err == nil || !strings.Contains(err.Error(), "process exited") {
		t.Fatalf("Start with a missing model = %v, want a report that the process exited", err)
	}
}

// TestRouterModeHidesLocalPaths starts a server in router mode on a directory holding the test model and checks
// that the model list it returns names no local directory.
func TestRouterModeHidesLocalPaths(t *testing.T) {
	model := harness.Model(t, "stories260K")
	dir := t.TempDir()
	if err := os.Symlink(model, filepath.Join(dir, filepath.Base(model))); err != nil {
		t.Fatal(err)
	}
	srv := harness.StartT(t, harness.Options{ModelsDir: dir})
	c, _ := srv.Client(t)
	models, err := llama.Call[jsontext.Value](t.Context(), c, http.MethodGet, "/models", nil)
	if err != nil {
		t.Fatalf("GET /models in router mode: %v", err)
	}
	if !strings.Contains(string(models), "stories260K") {
		t.Errorf("router lists no stories260K model: %s", models)
	}
	for _, local := range []string{dir, filepath.Dir(model), os.Getenv("HOME")} {
		if local != "" && strings.Contains(string(models), local) {
			t.Errorf("GET /models reveals the local path %s: %s", local, models)
		}
	}
}
