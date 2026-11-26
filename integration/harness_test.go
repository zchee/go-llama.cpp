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
	"errors"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
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

func TestServerCacheIsolation(t *testing.T) {
	inherited := t.TempDir()
	t.Setenv("LLAMA_CACHE", inherited)
	t.Setenv("HF_HUB_CACHE", inherited)
	// If the config is read, the router logs its path and requires credentials for GET /models.
	configEnv := "XDG_CONFIG_HOME"
	if runtime.GOOS == "windows" {
		configEnv = "APPDATA"
	}
	configHome := t.TempDir()
	t.Setenv(configEnv, configHome)
	inheritedConfig := filepath.Join(configHome, "llama.cpp", "config.ini")
	if err := os.Mkdir(filepath.Dir(inheritedConfig), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(inheritedConfig, []byte("api-key = must-not-be-read\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var servers []*harness.Server
	for _, env := range [][]string{nil, {"LLAMA_CACHE=" + inherited, configEnv + "=" + configHome}} {
		srv := harness.StartT(t, harness.Options{UnixSocket: true, Args: []string{"--verbosity", "4"}, Env: env})
		if strings.Contains(srv.LogTail(1<<20), "using config file: "+inheritedConfig) {
			t.Fatal("llama-server read the inherited user config file")
		}
		c, _ := srv.Client(t)
		if _, err := c.ListModels(t.Context()); err != nil {
			t.Fatalf("GET /models without credentials: %v (a config file likely set an api-key)", err)
		}
		servers = append(servers, srv)
		cache := srv.CacheDir()
		if seen[cache] || cache == inherited || filepath.Dir(cache) != filepath.Dir(srv.Socket) {
			t.Fatalf("cache %q is not unique to the server scratch directory %q", cache, filepath.Dir(srv.Socket))
		}
		seen[cache] = true
		// This trace comes from the router itself after reading its process environment, not the supervisor.
		if !strings.Contains(srv.LogTail(1<<20), "cached model presets from "+cache+"\n") {
			t.Fatal("llama-server did not resolve its cache to the private directory")
		}
		if info, err := os.Stat(cache); err != nil || !info.IsDir() || info.Mode().Perm()&0o077 != 0 {
			t.Fatalf("expected private cache directory: %v (%v)", info, err)
		}
	}
	for _, srv := range servers {
		if err := srv.Stop(); err != nil {
			t.Fatal(err)
		}
		for _, path := range []string{srv.CacheDir(), filepath.Dir(srv.Socket)} {
			if _, err := os.Stat(path); !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("%s was not removed by Stop: %v", path, err)
			}
		}
	}
	if files, err := os.ReadDir(inherited); err != nil || len(files) != 0 {
		t.Fatalf("inherited cache changed: %v (%v)", files, err)
	}
}

func TestServerSlotsDebugEnv(t *testing.T) {
	t.Setenv("LLAMA_SERVER_SLOTS_DEBUG", "0")
	srv := harness.StartT(t, harness.Options{
		Model: harness.Model(t, "stories260K"), Args: []string{"--ctx-size", "512", "--parallel", "1"},
		Env: []string{"LLAMA_SERVER_SLOTS_DEBUG=1"},
	})
	c, _ := srv.Client(t)
	const prompt = "Once"
	if _, err := c.Completion(t.Context(), &llama.CompletionRequest{Prompt: llama.TextPrompt(prompt), NPredict: new(1), IDSlot: new(0)}); err != nil {
		t.Fatal(err)
	}
	slots, err := c.Slots(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(slots) != 1 || slots[0].ID != 0 || slots[0].Prompt == nil || !strings.Contains(*slots[0].Prompt, prompt) {
		t.Errorf("GET /slots = %+v, want slot 0 with debug prompt containing %q", slots, prompt)
	}
}

func TestStartFailsFast(t *testing.T) {
	_, err := harness.Start(t.Context(), harness.Options{Model: filepath.Join(t.TempDir(), "missing.gguf")})
	if err == nil || !strings.Contains(err.Error(), "process exited") {
		t.Fatalf("Start with a missing model = %v, want a report that the process exited", err)
	}
	// The diagnostics are the server's own output, which names the model file it could not open.
	_, tail, _ := strings.Cut(err.Error(), "--- server log tail ---")
	if strings.Contains(tail, "log unavailable") || !strings.Contains(tail, "missing.gguf") {
		t.Errorf("the error lacks the server log tail:\n%s", err)
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

// TestObserverConcurrentReads reads the observed exchanges from another goroutine while a stream is in flight; run
// with -race, it fails if Exchanges hands out state that RoundTrip is still writing.
func TestObserverConcurrentReads(t *testing.T) {
	srv := presetStories260K.server(t)
	c, obs := srv.Client(t)
	stop := make(chan struct{})
	done := make(chan int)
	go func() {
		reads := 0
		for {
			select {
			case <-stop:
				done <- reads
				return
			default:
			}
			for _, ex := range obs.Exchanges() {
				_ = ex.StatusCode
				_ = ex.Err
				_ = ex.BodyClosed()
				_ = ex.BytesRead()
			}
			reads++
		}
	}()
	for range 3 {
		for _, err := range llama.Stream[streamChunk](t.Context(), c, http.MethodPost, "/completion", streamRequest) {
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	close(stop)
	if reads := <-done; reads == 0 {
		t.Errorf("the reader never observed the exchanges")
	}
	ex := obs.Exchanges()
	if len(ex) != 3 || ex[2].StatusCode != http.StatusOK || !ex[2].BodyClosed() {
		t.Errorf("exchanges after the streams: %+v", ex)
	}
}
