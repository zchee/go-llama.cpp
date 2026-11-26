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

// Give each contract its own server: the cache and model directory belong to the subtest.
func modelsPreset(name string, online bool) *serverPreset {
	return &serverPreset{name: "models-" + name, options: func(t testing.TB) harness.Options {
		dir := t.TempDir()
		if !online {
			if err := os.Symlink(harness.Model(t, "stories260K"), filepath.Join(dir, modelsLocalID+".gguf")); err != nil {
				t.Fatal(err)
			}
		}
		return harness.Options{ModelsDir: dir, Online: online, Args: []string{"--verbosity", "4", "--parallel", "1", "--ctx-size", "512", "--n-gpu-layers", "0"}}
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
	p := modelsPreset("events-and-errors", false)
	srv := harness.StartT(t, p.options(t))
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
	}
	for name, tt := range tests {
		t.Run(name, tt.run)
	}
}
