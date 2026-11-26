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

package harness

import "syscall"

// sysProcAttr isolates each process group from terminal interrupts and from signals to the other group.
func sysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}

func killDescendants(pid int) { killGroup(pid) }

// killGroup kills the process group led by pid.
func killGroup(pid int) {
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}
