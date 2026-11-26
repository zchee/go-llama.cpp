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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"testing"
	"time"

	yaml "go.yaml.in/yaml/v4"
)

// EnvModelCache names the environment variable that selects the model cache directory.
const EnvModelCache = "LLAMA_MODEL_CACHE"

// EnvModelsFile names the environment variable that selects the models file instead of integration/models.yaml.
const EnvModelsFile = "LLAMA_MODELS_FILE"

// downloadTimeout bounds one model download.
const downloadTimeout = 30 * time.Minute

// userAgent is the only identifying header sent to Hugging Face; requests carry no credentials or cookies.
const userAgent = "go-llama.cpp-integration"

// ModelSpec pins one model file.
type ModelSpec struct {
	// Name is the key of the entry in the models file.
	Name string `yaml:"-"`
	// Repo is the Hugging Face repository, for example "ggml-org-dev/test-model-stories260K".
	Repo string `yaml:"repo"`
	// Revision is the commit hash of the repository.
	Revision string `yaml:"revision"`
	// File is the path of the file in the repository.
	File string `yaml:"file"`
	// SHA256 is the hex sha256 of the file.
	SHA256 string `yaml:"sha256"`
	// Size is the file size in bytes.
	Size int64 `yaml:"size"`
}

var (
	revisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	sha256Pattern   = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// validate checks that the spec pins an exact file.
func (m ModelSpec) validate() error {
	switch {
	case m.Repo == "" || m.File == "":
		return fmt.Errorf("model %q: repo and file are required", m.Name)
	case !revisionPattern.MatchString(m.Revision):
		return fmt.Errorf("model %q: revision %q is not a commit hash", m.Name, m.Revision)
	case !sha256Pattern.MatchString(m.SHA256):
		return fmt.Errorf("model %q: sha256 %q is not a hex sha256", m.Name, m.SHA256)
	case m.Size <= 0:
		return fmt.Errorf("model %q: size must be positive", m.Name)
	}
	return nil
}

// URL returns the download URL of the pinned file.
func (m ModelSpec) URL() string {
	return "https://huggingface.co/" + m.Repo + "/resolve/" + m.Revision + "/" + (&url.URL{Path: m.File}).EscapedPath()
}

// ModelsFile returns the path of the models file: $LLAMA_MODELS_FILE, or models.yaml next to this package.
func ModelsFile() string {
	if p := os.Getenv(EnvModelsFile); p != "" {
		return p
	}
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "models.yaml"
	}
	return filepath.Join(filepath.Dir(file), "..", "models.yaml")
}

// LoadModels parses a models file.
func LoadModels(data []byte) (map[string]ModelSpec, error) {
	var doc struct {
		Models map[string]ModelSpec `yaml:"models"`
	}
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("harness: parsing models file: %w", err)
	}
	if len(doc.Models) == 0 {
		return nil, errors.New("harness: models file lists no models")
	}
	var errs []error
	for name, m := range doc.Models {
		m.Name = name
		doc.Models[name] = m
		errs = append(errs, m.validate())
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return doc.Models, nil
}

// ModelNames returns the sorted names of the models.
func ModelNames(models map[string]ModelSpec) []string {
	names := make([]string, 0, len(models))
	for n := range models {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// CacheDir returns the model cache directory: $LLAMA_MODEL_CACHE, or go-llama.cpp/models under the user cache
// directory.
func CacheDir() (string, error) {
	if d := os.Getenv(EnvModelCache); d != "" {
		return d, nil
	}
	d, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, "go-llama.cpp", "models"), nil
}

// AssetError reports that a model file could not be obtained, which is a problem of the test environment rather
// than a test failure.
type AssetError struct {
	Model string
	Err   error
}

func (e *AssetError) Error() string {
	return fmt.Sprintf("model asset %q unavailable: %v", e.Model, e.Err)
}

func (e *AssetError) Unwrap() error { return e.Err }

// Fetch returns the path of the model file in cacheDir, downloading it when it is missing. The file is verified
// against the pinned size and sha256 every time.
func Fetch(ctx context.Context, m ModelSpec, cacheDir string) (string, error) {
	path := filepath.Join(cacheDir, filepath.FromSlash(m.Repo), m.Revision, filepath.FromSlash(m.File))
	if err := verify(path, m); err == nil {
		return path, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", &AssetError{Model: m.Name, Err: fmt.Errorf("cached file %s: %w", path, err)}
	}
	if err := download(ctx, m, path); err != nil {
		return "", &AssetError{Model: m.Name, Err: err}
	}
	return path, nil
}

// verify checks the size and sha256 of the file at path.
func verify(path string, m ModelSpec) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return err
	}
	if n != m.Size {
		return fmt.Errorf("size %d, want %d", n, m.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.SHA256 {
		return fmt.Errorf("sha256 %s, want %s", got, m.SHA256)
	}
	return nil
}

// download fetches the pinned file into path through a temporary file, so that a failed or concurrent download
// never leaves a partial file under the final name.
func download(ctx context.Context, m ModelSpec, path string) error {
	ctx, cancel := context.WithTimeout(ctx, downloadTimeout)
	defer cancel()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.URL(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)
	client := &http.Client{Transport: http.DefaultTransport.(*http.Transport).Clone()}
	defer client.CloseIdleConnections()
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", m.URL(), resp.Status)
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), io.LimitReader(resp.Body, m.Size+1))
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("downloading %s: %w", m.URL(), err)
	}
	if n != m.Size {
		return fmt.Errorf("downloaded %d bytes from %s, want %d", n, m.URL(), m.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.SHA256 {
		return fmt.Errorf("sha256 of %s is %s, want %s", m.URL(), got, m.SHA256)
	}
	return os.Rename(tmp.Name(), path)
}

// Model returns the path of the named model of the models file, fetching it when needed. A model that cannot be
// obtained stops the test with a message that names it as an asset problem.
func Model(t testing.TB, name string) string {
	t.Helper()
	path, err := ModelPath(t.Context(), name)
	if err != nil {
		t.Fatal(err)
	}
	return path
}

// ModelPath is Model for code that is not a test, such as the recording tool.
func ModelPath(ctx context.Context, name string) (string, error) {
	data, err := os.ReadFile(ModelsFile())
	if err != nil {
		return "", &AssetError{Model: name, Err: err}
	}
	models, err := LoadModels(data)
	if err != nil {
		return "", &AssetError{Model: name, Err: err}
	}
	m, ok := models[name]
	if !ok {
		return "", &AssetError{Model: name, Err: fmt.Errorf("not listed in %s (listed: %v)", ModelsFile(), ModelNames(models))}
	}
	dir, err := CacheDir()
	if err != nil {
		return "", &AssetError{Model: name, Err: err}
	}
	return Fetch(ctx, m, dir)
}
