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
	"errors"
	"sync"
	"testing"

	"github.com/zchee/go-llama.cpp/integration/harness"
)

// serverPreset is a server configuration shared by the tests that need it. Each preset starts at most one
// process per test binary run; TestMain stops them all at the end.
type serverPreset struct {
	// name identifies the preset.
	name string
	// options returns the options of the server; it may fetch model files with harness.Model.
	options func(t testing.TB) harness.Options
}

// presetStories260K is the upstream "tinyllama2" test preset: a tiny text model for completion, tokenization,
// slots, props and the OpenAI-compatible text endpoints.
var presetStories260K = &serverPreset{
	name: "stories260K",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "stories260K"),
			Args: []string{
				"--alias", "tinyllama-2",
				"--ctx-size", "512",
				"--batch-size", "32",
				"--parallel", "2",
				"--n-predict", "64",
				"--seed", "42",
			},
		}
	},
}

var (
	sharedMu      sync.Mutex
	sharedServers = map[string]*harness.Server{}
)

// server returns the running server of the preset, starting it on first use.
func (p *serverPreset) server(t testing.TB) *harness.Server {
	t.Helper()
	sharedMu.Lock()
	defer sharedMu.Unlock()
	if s, ok := sharedServers[p.name]; ok {
		if s.Exited() {
			t.Fatalf("shared server %q has exited; log tail:\n%s", p.name, s.LogTail(4096))
		}
		return s
	}
	// The context only bounds the wait for readiness; the process outlives the test that started it.
	s, err := harness.Start(t.Context(), p.options(t))
	if err != nil {
		t.Fatalf("starting shared server %q: %v", p.name, err)
	}
	sharedServers[p.name] = s
	return s
}

// stopSharedServers stops every shared server.
func stopSharedServers() error {
	sharedMu.Lock()
	defer sharedMu.Unlock()
	var errs []error
	for name, s := range sharedServers {
		if err := s.Stop(); err != nil {
			errs = append(errs, err)
		}
		delete(sharedServers, name)
	}
	return errors.Join(errs...)
}
