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

package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestModelsFile(t *testing.T) {
	data, err := os.ReadFile(ModelsFile())
	if err != nil {
		t.Fatalf("reading %s: %v", ModelsFile(), err)
	}
	models, err := LoadModels(data)
	if err != nil {
		t.Fatalf("the checked-in models file is invalid: %v", err)
	}
	m, ok := models["stories260K"]
	if !ok {
		t.Fatalf("models file lacks stories260K: %v", ModelNames(models))
	}
	const wantURL = "https://huggingface.co/ggml-org-dev/test-model-stories260K/resolve/0724641ef52d46f564590e6121ceaa22dc97e5d0/stories260K-f32.gguf"
	if m.URL() != wantURL {
		t.Errorf("URL = %s, want %s", m.URL(), wantURL)
	}
}

func TestLoadModelsRejectsLoosePins(t *testing.T) {
	const good = "repo: a/b\n    revision: 0724641ef52d46f564590e6121ceaa22dc97e5d0\n    file: m.gguf\n    sha256: 270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d\n    size: 1\n"
	tests := map[string]struct {
		doc  string
		want string
	}{
		"success: pinned":         {doc: "models:\n  m:\n    " + good},
		"error: branch revision":  {doc: "models:\n  m:\n    " + strings.Replace(good, "0724641ef52d46f564590e6121ceaa22dc97e5d0", "main", 1), want: "not a commit hash"},
		"error: short sha256":     {doc: "models:\n  m:\n    " + strings.Replace(good, "270cba1b", "", 1), want: "not a hex sha256"},
		"error: no file":          {doc: "models:\n  m:\n    " + strings.Replace(good, "file: m.gguf", "file: ''", 1), want: "repo and file are required"},
		"error: zero size":        {doc: "models:\n  m:\n    " + strings.Replace(good, "size: 1", "size: 0", 1), want: "size must be positive"},
		"error: empty":            {doc: "models: {}\n", want: "lists no models"},
		"error: not YAML mapping": {doc: "models: [1, 2]\n", want: "parsing models file"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := LoadModels([]byte(tt.doc))
			if tt.want == "" {
				if err != nil {
					t.Fatalf("LoadModels: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("LoadModels error = %v, want one containing %q", err, tt.want)
			}
		})
	}
}

// TestFetchVerifiesCachedFiles places files in a cache directory and checks that only an exact copy is used. A
// file that differs is reported as an asset problem instead of being replaced silently.
func TestFetchVerifiesCachedFiles(t *testing.T) {
	content := []byte("GGUF model bytes")
	sum := sha256.Sum256(content)
	spec := ModelSpec{
		Name: "m", Repo: "org/repo", Revision: "0724641ef52d46f564590e6121ceaa22dc97e5d0",
		File: "dir/m.gguf", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(content)),
	}
	tests := map[string]struct {
		content []byte
		wantErr string
	}{
		"success: exact copy":    {content: content},
		"error: different bytes": {content: []byte("GGUF model bytez"), wantErr: "sha256"},
		"error: truncated":       {content: content[:4], wantErr: "size 4"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			cache := t.TempDir()
			path := filepath.Join(cache, "org", "repo", spec.Revision, "dir", "m.gguf")
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, tt.content, 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := Fetch(t.Context(), spec, cache)
			if tt.wantErr == "" {
				if err != nil || got != path {
					t.Fatalf("Fetch = %q, %v; want %q", got, err, path)
				}
				return
			}
			assetErr, ok := errors.AsType[*AssetError](err)
			if !ok || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Fetch error = %v, want an *AssetError containing %q", err, tt.wantErr)
			}
			if assetErr.Model != "m" {
				t.Errorf("AssetError.Model = %q", assetErr.Model)
			}
		})
	}
}
