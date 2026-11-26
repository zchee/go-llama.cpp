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
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

const (
	modelsLocalID = "stories260K-f32"
	modelsHubID   = "ggml-org-dev/test-model-stories260K:F32"
)

var (
	modelsDownloadPreset = modelsPreset("download", true)
	modelsDeletePreset   = modelsPreset("delete", true)
)

// modelsOptions returns router-mode options whose model directory is dir.
func modelsOptions(t testing.TB, dir string, online bool) harness.Options {
	if !online {
		if err := os.Symlink(harness.Model(t, "stories260K"), filepath.Join(dir, modelsLocalID+".gguf")); err != nil {
			t.Fatal(err)
		}
	}
	return harness.Options{ModelsDir: dir, Online: online, Args: []string{"--verbosity", "4", "--parallel", "1", "--ctx-size", "512", "--n-gpu-layers", "0"}}
}

// modelsPreset returns a shared router preset whose models directory outlives the subtest that starts it.
func modelsPreset(name string, online bool) *serverPreset {
	return &serverPreset{name: "models-" + name, options: func(t testing.TB) harness.Options {
		dir, err := os.MkdirTemp("", "go-llama-models-"+name+"-*")
		if err != nil {
			t.Fatal(err)
		}
		registerSharedDir(dir)
		return modelsOptions(t, dir, online)
	}}
}

// Verify the router actually selected the new empty cache before allowing any mutating HTTP request.
func modelsCheckCache(t *testing.T, srv *harness.Server) {
	t.Helper()
	cache := srv.CacheDir()
	if !filepath.IsAbs(cache) || !strings.Contains(srv.LogTail(1<<20), "cached model presets from "+cache+"\n") {
		t.Fatal("router has not reported the isolated cache; refusing to mutate it")
	}
	files, err := os.ReadDir(cache)
	if err != nil || len(files) != 0 {
		t.Fatalf("cache is not empty before download: %v (%v)", files, err)
	}
}

func modelsWait(ctx context.Context, c *llama.Client, id string, state llama.ModelState) (*llama.Model, error) {
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		list, err := c.ListModels(ctx)
		if err != nil {
			return nil, err
		}
		for i := range list.Data {
			m := &list.Data[i]
			if m.ID == id && m.Status != nil && m.Status.Value == state {
				return m, nil
			}
		}
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("waiting for %q to be %q: %w", id, state, ctx.Err())
		case <-tick.C:
		}
	}
}

func modelsDownloadAndCheck(t *testing.T, c *llama.Client, srv *harness.Server) string {
	t.Helper()
	modelsCheckCache(t, srv)
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	res, err := c.DownloadModel(ctx, &llama.DownloadModelRequest{Model: modelsHubID, Sidecars: []string{}})
	if err != nil || !res.Success {
		t.Fatalf("starting download: %v (%v)", res, err)
	}
	m, err := modelsWait(ctx, c, modelsHubID, llama.ModelUnloaded)
	if err != nil {
		t.Fatal(err)
	}
	if m.Source != "cache" || m.CanRemove == nil || !*m.CanRemove {
		t.Fatalf("not a removable cached model: %+v", m)
	}
	// Open through a confined root: the HF snapshot entry may be a relative symlink into blobs.
	cache := srv.CacheDir()
	root, err := os.OpenRoot(cache)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	files, err := fs.Glob(root.FS(), "models--*/snapshots/*/*.gguf")
	if err != nil || len(files) != 1 {
		t.Fatalf("GGUF files in isolated cache = %v (%v)", files, err)
	}
	file, err := root.Open(filepath.FromSlash(files[0]))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		t.Fatal(err)
	}
	pinned, err := os.Open(harness.Model(t, "stories260K"))
	if err != nil {
		t.Fatal(err)
	}
	defer pinned.Close()
	want := sha256.New()
	if _, err := io.Copy(want, pinned); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(h.Sum(nil), want.Sum(nil)) {
		t.Fatal("download did not match the pinned model; refusing deletion")
	}
	return filepath.Join(cache, filepath.FromSlash(files[0]))
}

func modelsCheckDeleted(path string) error {
	_, err := os.Lstat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	return fmt.Errorf("downloaded snapshot still present after deletion: %s", filepath.Base(path))
}

func TestModelsRouterEventsAndErrors(t *testing.T) {
	srv := harness.StartT(t, modelsOptions(t, t.TempDir(), false))
	c, obs := srv.Client(t)
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	tests := map[string]struct{ run func(*testing.T) }{
		"success: reload event and early stream close": {func(t *testing.T) {
			sub, stop := context.WithCancel(ctx)
			defer stop()
			done := make(chan error, 1)
			go func() {
				result := errors.New("missing reload event")
				for ev, err := range c.ModelEvents(sub) {
					result = err
					if err == nil && (ev.Model != "*" || ev.Event != "models_reload") {
						result = fmt.Errorf("unexpected reload event: %+v", ev)
					}
					break
				}
				done <- result
			}()
			tick := time.NewTicker(50 * time.Millisecond)
			defer tick.Stop()
			for {
				select {
				case err := <-done:
					if err != nil {
						t.Fatal(err)
					}
					for _, ex := range obs.Exchanges() {
						if ex.Path == "/models/sse" && ex.BodyClosed() {
							return
						}
					}
					t.Fatal("breaking the iterator did not close its response")
				case <-tick.C:
					if _, err := c.ListModels(ctx, llama.WithReload()); err != nil {
						t.Fatal(err)
					}
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				}
			}
		}},
		"error: missing load target": {func(t *testing.T) {
			_, err := c.LoadModel(ctx, &llama.ModelRequest{Model: "unknown"})
			api, ok := errors.AsType[*llama.APIError](err)
			if !ok || api.StatusCode != http.StatusNotFound {
				t.Errorf("error = %v, want HTTP 404", err)
			}
		}},
		"error: unloading an idle model": {func(t *testing.T) {
			_, err := c.UnloadModel(ctx, &llama.ModelRequest{Model: modelsLocalID})
			api, ok := errors.AsType[*llama.APIError](err)
			if !ok || api.StatusCode != http.StatusBadRequest {
				t.Errorf("error = %v, want HTTP 400", err)
			}
		}},
		"error: empty model name on delete": {func(t *testing.T) {
			dc, dobs := srv.Client(t)
			_, err := dc.DeleteModel(ctx, "")
			if err == nil || !strings.Contains(err.Error(), "non-empty model name") {
				t.Errorf("error = %v, want client-side non-empty model name error", err)
			}
			if ex := dobs.Exchanges(); len(ex) != 0 {
				t.Fatalf("empty model deletion sent %d HTTP request(s): %+v", len(ex), ex)
			}
		}},
	}
	for name, tt := range tests {
		t.Run(name, tt.run)
	}
}

func TestModelsSharedPresetDirectorySurvivesSubtest(t *testing.T) {
	preset := modelsPreset("listing", false)
	t.Run("success: start shared preset", func(t *testing.T) {
		srv := preset.server(t)
		dir := filepath.Join(filepath.Dir(srv.CacheDir()), "models-dir")
		if info, err := os.Stat(dir); err != nil || !info.IsDir() {
			t.Fatalf("models directory missing right after start: %v (%v)", info, err)
		}
	})
	t.Run("success: directory and model survive after starting subtest ends", func(t *testing.T) {
		srv := preset.server(t)
		dir := filepath.Join(filepath.Dir(srv.CacheDir()), "models-dir")
		if info, err := os.Stat(dir); err != nil || !info.IsDir() {
			t.Fatalf("models directory disappeared while shared server is running: %v (%v)", info, err)
		}
		if _, err := os.Stat(filepath.Join(dir, modelsLocalID+".gguf")); err != nil {
			t.Fatalf("model symlink disappeared while shared server is running: %v", err)
		}
		c, _ := srv.Client(t)
		list, err := c.ListModels(t.Context(), llama.WithReload())
		if err != nil {
			t.Fatalf("reloading models on reused preset: %v", err)
		}
		if len(list.Data) != 1 || list.Data[0].ID != modelsLocalID {
			t.Fatalf("reused preset lost model %q: %+v", modelsLocalID, list)
		}
	})
}

func TestModelsSharedPresetCleanup(t *testing.T) {
	dir, err := os.MkdirTemp("", "go-llama-models-cleanup-*")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "marker"), []byte("x"), 0o600); err != nil {
		_ = os.RemoveAll(dir)
		t.Fatal(err)
	}

	sharedMu.Lock()
	savedServers := sharedServers
	sharedServers = map[string]*harness.Server{}
	sharedMu.Unlock()

	sharedDirsMu.Lock()
	savedDirs := sharedDirs
	sharedDirs = nil
	sharedDirsMu.Unlock()

	t.Cleanup(func() {
		sharedMu.Lock()
		sharedServers = savedServers
		sharedMu.Unlock()
		sharedDirsMu.Lock()
		sharedDirs = savedDirs
		sharedDirsMu.Unlock()
		_ = os.RemoveAll(dir)
	})

	registerSharedDir(dir)
	if err := stopSharedServers(); err != nil {
		t.Fatalf("stopSharedServers: %v", err)
	}
	if _, err := os.Stat(dir); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("registered preset directory %q still exists after stopSharedServers: %v", dir, err)
	}
	sharedDirsMu.Lock()
	remaining := len(sharedDirs)
	sharedDirsMu.Unlock()
	if remaining != 0 {
		t.Errorf("sharedDirs has %d entries after stopSharedServers, want 0", remaining)
	}
}
