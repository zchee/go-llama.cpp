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

package llama_test

import (
	"context"
	"fmt"

	llama "github.com/zchee/go-llama.cpp"
)

func ExampleClient_CompletionStream() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	req := &llama.CompletionRequest{
		Prompt:   llama.TextPrompt("Once upon a time"),
		NPredict: new(16),
	}
	for chunk, err := range c.CompletionStream(context.Background(), req) {
		if err != nil {
			panic(err)
		}
		if chunk.Content != nil {
			fmt.Print(*chunk.Content)
		}
	}
}

func ExampleClient_LoadModel() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	ctx := context.Background()
	if _, err := c.LoadModel(ctx, &llama.ModelRequest{Model: "qwen2.5-0.5b-instruct"}); err != nil {
		panic(err)
	}
	list, err := c.ListModels(ctx)
	if err != nil {
		panic(err)
	}
	for i := range list.Data {
		m := &list.Data[i]
		if m.Status != nil {
			fmt.Println(m.ID, m.Status.Value)
		}
	}
}
