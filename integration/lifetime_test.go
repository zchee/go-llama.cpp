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

//go:build unix

package integration

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"github.com/zchee/go-llama.cpp/integration/harness"
)

// Environment of the child process of TestServersDieWithTheTestProcess.
const (
	envOwnerMode    = "GO_LLAMA_TEST_OWNER_MODE"
	envOwnerPIDFile = "GO_LLAMA_TEST_OWNER_PID_FILE"
)

// TestServersDieWithTheTestProcess runs this test binary again as a child that starts a server and then either
// panics or is killed with SIGKILL, and checks that the server of that child does not outlive it.
func TestServersDieWithTheTestProcess(t *testing.T) {
	tests := map[string]string{
		"success: owner panics":    "panic",
		"success: owner is killed": "kill",
	}
	for name, mode := range tests {
		t.Run(name, func(t *testing.T) {
			pidPath := filepath.Join(t.TempDir(), "server.pid")
			var out bytes.Buffer
			cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestHelperServerOwner$", "-test.count=1", "-test.v")
			cmd.Env = append(os.Environ(), envOwnerMode+"="+mode, envOwnerPIDFile+"="+pidPath)
			cmd.Stdout = &out
			cmd.Stderr = &out
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			exited := make(chan error, 1)
			go func() { exited <- cmd.Wait() }()
			pid, err := waitForPID(pidPath, exited, 2*time.Minute)
			if err != nil {
				_ = cmd.Process.Kill()
				<-exited
				t.Fatalf("the child did not start a server: %v\n%s", err, out.String())
			}
			if !alive(pid) {
				t.Fatalf("llama-server %d is not running while its owner is", pid)
			}
			if mode == "kill" {
				if err := cmd.Process.Signal(syscall.SIGKILL); err != nil {
					t.Fatal(err)
				}
			}
			err = <-exited
			if _, ok := errors.AsType[*exec.ExitError](err); !ok {
				t.Fatalf("the child ended with %v, want a failed exit\n%s", err, out.String())
			}
			deadline := time.Now().Add(30 * time.Second)
			for alive(pid) {
				if time.Now().After(deadline) {
					_ = syscall.Kill(pid, syscall.SIGKILL)
					t.Fatalf("llama-server %d outlived its owner by 30s (child output:\n%s)", pid, out.String())
				}
				time.Sleep(50 * time.Millisecond)
			}
		})
	}
}

// TestHelperServerOwner is the child process of TestServersDieWithTheTestProcess. It does nothing unless that test
// started it.
func TestHelperServerOwner(t *testing.T) {
	mode := os.Getenv(envOwnerMode)
	if mode == "" {
		return
	}
	srv, err := harness.Start(t.Context(), harness.Options{
		Model: harness.Model(t, "stories260K"),
		Args:  []string{"--ctx-size", "256", "--parallel", "1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	path := os.Getenv(envOwnerPIDFile)
	if err := os.WriteFile(path+".tmp", []byte(strconv.Itoa(srv.PID())), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path+".tmp", path); err != nil {
		t.Fatal(err)
	}
	switch mode {
	case "panic":
		panic("owner of a running llama-server panics")
	case "kill":
		time.Sleep(10 * time.Minute)
	}
}

// waitForPID waits until the child writes the server PID. A child that exits first puts its result back into
// exited for the caller.
func waitForPID(path string, exited chan error, timeout time.Duration) (int, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			return strconv.Atoi(string(data))
		}
		select {
		case err := <-exited:
			exited <- err
			// The child may write the file and exit between two polls.
			if data, readErr := os.ReadFile(path); readErr == nil {
				return strconv.Atoi(string(data))
			}
			return 0, fmt.Errorf("the child exited first: %w", err)
		case <-time.After(50 * time.Millisecond):
		}
	}
	return 0, errors.New("timed out")
}

// alive reports whether a process with the PID exists. A process that exited and was reaped is gone; the
// supervisor reaps llama-server before it exits itself.
func alive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}
