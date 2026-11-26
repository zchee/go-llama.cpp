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
	"fmt"
	"os"
	"testing"

	"go.uber.org/goleak"
)

// envIntegration enables the tests of this package.
const envIntegration = "LLAMA_INTEGRATION"

func TestMain(m *testing.M) {
	if os.Getenv(envIntegration) != "1" {
		fmt.Fprintf(os.Stderr, "integration tests skipped: set %s=1 to run them against a real llama-server\n", envIntegration)
		os.Exit(0)
	}
	goleak.VerifyTestMain(runThenStopServers{m})
}

// runThenStopServers runs the tests and then stops the shared servers, so that the leak check that follows sees
// no goroutine waiting on a server process.
type runThenStopServers struct {
	m *testing.M
}

func (r runThenStopServers) Run() int {
	code := r.m.Run()
	if err := stopSharedServers(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		if code == 0 {
			code = 1
		}
	}
	return code
}
