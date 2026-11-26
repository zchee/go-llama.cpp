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
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
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
	tests := map[string]struct{ mode string }{
		"success: owner panics":    {"panic"},
		"success: owner is killed": {"kill"},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			testOwnerExit(t, tt.mode)
		})
	}
}

func TestStuckRouterDiesWithItsOwner(t *testing.T) {
	testOwnerExit(t, "router")
}

func testOwnerExit(t *testing.T, mode string) {
	t.Helper()
	pidPath := filepath.Join(t.TempDir(), "server.pid")
	var out bytes.Buffer
	cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestHelperServerOwner$", "-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), envOwnerMode+"="+mode, envOwnerPIDFile+"="+pidPath)
	cmd.Stdout = &out
	cmd.Stderr = &out
	ack, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer ack.Close()
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
	pids := []int{pid}
	if mode == "router" {
		// Snapshot the loaded router's tree before orphaning it; parent PIDs change on exit.
		data, err := exec.CommandContext(t.Context(), "ps", "-A", "-o", "pid=,ppid=").Output()
		if err != nil {
			t.Fatal(err)
		}
		children := map[int][]int{}
		for line := range strings.Lines(string(data)) {
			var child, parent int
			if _, err := fmt.Sscan(line, &child, &parent); err != nil {
				t.Fatal(err)
			}
			children[parent] = append(children[parent], child)
		}
		for i := 0; i < len(pids); i++ {
			pids = append(pids, children[pids[i]]...)
		}
		if len(pids) < 2 {
			t.Fatal("loaded router has no model process")
		}
	}
	for _, p := range pids {
		if !alive(p) {
			t.Fatalf("llama-server %d is not running while its owner is", p)
		}
		if mode == "router" {
			if group, err := syscall.Getpgid(p); err != nil || group != pid {
				t.Errorf("process %d group = %d, %v; want router group %d", p, group, err, pid)
			}
			if err := syscall.Kill(p, syscall.SIGSTOP); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if alive(p) {
					_ = syscall.Kill(p, syscall.SIGKILL)
				}
			})
		}
	}
	if mode == "panic" {
		if _, err := ack.Write([]byte{'!'}); err != nil {
			t.Fatal(err)
		}
	} else if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	err = <-exited
	if _, ok := errors.AsType[*exec.ExitError](err); !ok {
		t.Fatalf("the child ended with %v, want a failed exit\n%s", err, out.String())
	}
	if mode == "panic" && !strings.Contains(out.String(), "panic: owner acknowledged !") {
		t.Fatalf("child did not panic after acknowledgement:\n%s", out.String())
	}
	deadline := time.Now().Add(30 * time.Second)
	for _, p := range pids {
		for alive(p) {
			if time.Now().After(deadline) {
				_ = syscall.Kill(p, syscall.SIGKILL)
				t.Fatalf("llama-server %d outlived its owner (child output:\n%s)", p, out.String())
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
}

// TestHelperServerOwner is the child process of TestServersDieWithTheTestProcess. It does nothing unless that test
// started it.
func TestHelperServerOwner(t *testing.T) {
	mode := os.Getenv(envOwnerMode)
	if mode == "" {
		return
	}
	path := os.Getenv(envOwnerPIDFile)
	opts := harness.Options{Model: harness.Model(t, "stories260K"), Args: []string{"--ctx-size", "256", "--parallel", "1"}}
	if mode == "router" {
		opts.ModelsDir = filepath.Dir(path)
		if err := os.Symlink(opts.Model, filepath.Join(opts.ModelsDir, "tiny.gguf")); err != nil {
			t.Fatal(err)
		}
		opts.Model = ""
	}
	srv, err := harness.Start(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if mode == "router" {
		c, _ := srv.Client(t)
		if _, err := llama.Call[struct{}](t.Context(), c, http.MethodPost, "/models/load", map[string]string{"model": "tiny"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path+".tmp", []byte(strconv.Itoa(srv.PID())), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path+".tmp", path); err != nil {
		t.Fatal(err)
	}
	switch mode {
	case "panic":
		var ack [1]byte
		if _, err := os.Stdin.Read(ack[:]); err != nil {
			t.Fatal(err)
		}
		panic(fmt.Sprintf("owner acknowledged %c", ack[0]))
	case "kill", "router":
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
