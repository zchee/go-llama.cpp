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

package openai_test

import (
	"context"
	"fmt"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

func ExampleClient_ChatCompletions_tool() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	type weatherArgs struct {
		City string `json:"city"`
	}
	params, err := llama.SchemaFor[weatherArgs]()
	if err != nil {
		panic(err)
	}
	res, err := openai.NewClient(c).ChatCompletions(context.Background(), &openai.ChatCompletionRequest{
		Messages: []openai.Message{{Role: "user", Content: openai.TextContent("What is the weather in Kyoto?")}},
		Tools: []openai.Tool{{Type: "function", Function: &openai.FunctionDefinition{
			Name:        "weather",
			Description: new("Get the weather for a city."),
			Parameters:  params,
		}}},
		ToolChoice: openai.ToolChoiceFunction("weather"),
		MaxTokens:  new(128),
	})
	if err != nil {
		panic(err)
	}
	for i := range res.Choices {
		for _, call := range res.Choices[i].Message.ToolCalls {
			if f := call.Function; f != nil && f.Name != nil && f.Arguments != nil {
				fmt.Printf("%s: %s\n", *f.Name, *f.Arguments)
			}
		}
	}
}

func ExampleClient_Embeddings_float() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	resp, err := openai.NewClient(c).Embeddings(context.Background(), &openai.EmbeddingRequest{
		Input: llama.EmbeddingTexts("first sentence", "second sentence"),
	})
	if err != nil {
		panic(err)
	}
	for i := range resp.Data {
		fmt.Println(resp.Data[i].Index, resp.Data[i].Embedding.Len())
	}
}
