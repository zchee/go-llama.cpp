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
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"time"
)

// envSupervise marks a process that Start launched as the supervisor of one llama-server.
const envSupervise = "GO_LLAMA_HARNESS_SUPERVISE"

// pidFile is the file, in the working directory of the supervisor, that holds the PID of llama-server.
const pidFile = "server.pid"

// exitKilled is the exit status of a supervisor that had to kill llama-server after an interrupt.
const exitKilled = 70

// A llama-server started by Start must not outlive the process that started it, whatever ends that process: a
// test that panics, a go test timeout, or SIGKILL. Linux could use a parent-death signal, but macOS has none, and
// no exit hook runs on SIGKILL. What works on both is a pipe: Start runs a small supervisor (this binary again,
// recognised by envSupervise before main runs) that starts llama-server and reads its standard input, a pipe whose
// only write end is held by the starting process. The kernel closes that end when the process exits for any
// reason, the read returns EOF, and the supervisor interrupts llama-server, kills it after stopTimeout, removes
// the server's directory and exits. Stop uses the same path by closing the write end itself.
func init() {
	if os.Getenv(envSupervise) != "1" {
		return
	}
	os.Exit(supervise(os.Args[1:]))
}

// supervise runs argv as llama-server until it exits or standard input reaches EOF. It returns the exit status
// of llama-server when it exited on its own, 0 after a stop it obeyed, and exitKilled when it had to be killed.
func supervise(argv []string) int {
	// The supervisor runs in its own process group; an interrupt that still reaches it must not end it before
	// the server it guards.
	signal.Ignore(os.Interrupt)
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "harness supervisor: no command")
		return 2
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.SysProcAttr = sysProcAttr()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = slices.DeleteFunc(os.Environ(), func(kv string) bool { return strings.HasPrefix(kv, envSupervise+"=") })
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "harness supervisor: starting %s: %v\n", argv[0], err)
		return 127
	}
	// A router can exit before its model processes. Clean up any remaining group members on every exit.
	defer killDescendants(cmd.Process.Pid)
	if err := writePID(cmd.Process.Pid); err != nil {
		fmt.Fprintf(os.Stderr, "harness supervisor: %v\n", err)
	}

	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	ownerGone := make(chan struct{})
	go func() {
		_, _ = io.Copy(io.Discard, os.Stdin)
		close(ownerGone)
	}()

	select {
	case err := <-exited:
		if ee, ok := errors.AsType[*exec.ExitError](err); ok && ee.ExitCode() >= 0 {
			return ee.ExitCode()
		}
		if err != nil {
			return 1
		}
		return 0
	case <-ownerGone:
	}

	code := 0
	_ = cmd.Process.Signal(os.Interrupt)
	timer := time.NewTimer(stopTimeout)
	select {
	case <-exited:
		timer.Stop()
	case <-timer.C:
		killGroup(cmd.Process.Pid)
		<-exited
		code = exitKilled
	}
	// Nobody else will clean up after an owner that died; after Stop the owner removes the directory as well.
	if dir, err := os.Getwd(); err == nil {
		_ = os.RemoveAll(dir)
	}
	return code
}

// writePID writes the PID of llama-server so that the owner can find it; the rename makes the file appear whole.
func writePID(pid int) error {
	tmp := pidFile + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(pid)), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, pidFile)
}
