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

// Package integration runs the client against a real llama-server. The tests run only when LLAMA_INTEGRATION=1;
// the server binary comes from LLAMA_SERVER_BIN or PATH and the model files from integration/models.yaml.
//
// The contract test checks that every endpoint the server README documents has a typed method that reaches the
// documented route. Each endpoint group keeps its contract cases in its own contract_<group>_test.go file.
package integration
