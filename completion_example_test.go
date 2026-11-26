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
	"log"

	llama "github.com/zchee/go-llama.cpp"
)

func ExampleClient_Completion() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	r, err := c.Completion(context.Background(), &llama.CompletionRequest{Prompt: llama.TextPrompt("Once upon a time"), NPredict: new(16)})
	if err != nil {
		log.Fatal(err)
	}
	if r.Content != nil {
		fmt.Println(*r.Content)
	}
}

func ExampleClient_InfillStream() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	req := &llama.InfillRequest{InputPrefix: llama.InfillInput{Text: "func add(x, y int) int {"}, InputSuffix: llama.InfillInput{Text: "}"}, NPredict: new(16)}
	for chunk, err := range c.InfillStream(context.Background(), req) {
		if err != nil {
			log.Fatal(err)
		}
		if chunk.Content != nil {
			fmt.Print(*chunk.Content)
		}
	}
}

func ExampleClient_Props() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	r, err := c.Props(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(r.ModelAlias, r.DefaultGenerationSettings.NCtx)
}

func ExampleClient_Slots() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		log.Fatal(err)
	}
	slots, err := c.Slots(context.Background(), llama.WithFailOnNoSlot())
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range slots {
		fmt.Println(s.ID, s.IsProcessing)
	}
}
