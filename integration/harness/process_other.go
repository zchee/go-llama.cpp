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

//go:build !unix

package harness

import (
	"os"
	"syscall"
)

// sysProcAttr returns no attributes where process groups are not available.
func sysProcAttr() *syscall.SysProcAttr {
	return nil
}

// The process was already reaped; without process groups its descendants cannot be addressed.
func killDescendants(int) {}

// killGroup kills the immediate process where process groups are not available.
func killGroup(pid int) {
	if p, err := os.FindProcess(pid); err == nil {
		_ = p.Kill()
	}
}
