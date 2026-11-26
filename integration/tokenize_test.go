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
	"cmp"
	"encoding/json/jsontext"
	"errors"
	"net/http"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

var presetTokenizeJinja = &serverPreset{
	name: "tokenize-jinja",
	options: func(t testing.TB) harness.Options {
		opts := presetStories260K.options(t)
		opts.Args = append(opts.Args, "--jinja")
		return opts
	},
}

func TestTokenizePiecesAndOptions(t *testing.T) {
	srv := presetStories260K.server(t)
	tests := map[string]struct {
		request llama.TokenizeRequest
		ids     []llama.Token
		pieces  bool
	}{
		"success: invalid UTF-8 pieces": {
			request: llama.TokenizeRequest{Content: llama.TokenizeText("á"), WithPieces: new(true)},
			ids:     []llama.Token{410, 198, 164},
			pieces:  true,
		},
		"success: explicit false parses special tokens literally": {
			request: llama.TokenizeRequest{Content: llama.TokenizeText("<s>"), ParseSpecial: new(false), WithPieces: new(false)},
			ids:     []llama.Token{410, 504, 419, 505},
		},
		"success: default parse special returns BOS": {
			request: llama.TokenizeRequest{Content: llama.TokenizeText("<s>")},
			ids:     []llama.Token{1},
		},
		"success: add special to the first text segment": {
			request: llama.TokenizeRequest{Content: llama.TokenizeParts(llama.TextPart("Hello"), llama.TokenPart(0), llama.TextPart("!")), AddSpecial: new(true)},
			ids:     []llama.Token{1, 346, 306, 414, 0, 410, 443},
		},
		"success: do not add special if a token ID comes first": {
			request: llama.TokenizeRequest{Content: llama.TokenizeParts(llama.TokenPart(0), llama.TextPart("Hello")), AddSpecial: new(true)},
			ids:     []llama.Token{0, 346, 306, 414},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, observer := srv.Client(t)
			res, err := c.Tokenize(t.Context(), &tt.request, llama.WithModelQuery("tinyllama-2"))
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]llama.Token, len(res.Tokens))
			for i, token := range res.Tokens {
				ids[i] = token.ID
				if (token.Piece != nil) != tt.pieces {
					t.Fatalf("token %d has piece %v; want piece = %t", i, token.Piece, tt.pieces)
				}
			}
			if diff := gocmp.Diff(tt.ids, ids); diff != "" {
				t.Fatalf("token IDs differ (-want +got):\n%s", diff)
			}
			if tt.pieces {
				space, text := res.Tokens[0].Piece.Text()
				if space != " " || !text {
					t.Errorf("first piece = (%q, %t), want (%q, true)", space, text, " ")
				}
				bytes := append([]byte{}, res.Tokens[1].Piece.Bytes()...)
				bytes = append(bytes, res.Tokens[2].Piece.Bytes()...)
				if diff := gocmp.Diff([]byte{0xc3, 0xa1}, bytes); diff != "" {
					t.Errorf("byte pieces differ (-want +got):\n%s", diff)
				}
				detok, err := c.Detokenize(t.Context(), &llama.DetokenizeRequest{Tokens: ids[1:]})
				if err != nil {
					t.Fatal(err)
				}
				if detok.Content != "á" {
					t.Errorf("detokenize content = %q, want %q", detok.Content, "á")
				}
			}
			exchanges := observer.Exchanges()
			if len(exchanges) == 0 || exchanges[0].Query.Get("model") != "tinyllama-2" {
				t.Errorf("expected model query option, got %+v", exchanges)
			}
		})
	}
}

func TestApplyTemplateOptions(t *testing.T) {
	user := llama.Message{Role: "user", Content: llama.PartsContent(llama.TextContentPart("Say hi"))}
	assistant := llama.Message{Role: "assistant", Content: llama.TextContent("Hi")}
	tests := map[string]struct {
		request llama.ApplyTemplateRequest
		prompt  string
		preset  *serverPreset
	}{
		"success: render without a generation prompt": {
			request: llama.ApplyTemplateRequest{Messages: []llama.Message{user}, AddGenerationPrompt: new(false)},
			prompt:  "<|im_start|>user\nSay hi<|im_end|>\n",
		},
		"success: continue the final assistant message": {
			request: llama.ApplyTemplateRequest{Messages: []llama.Message{user, assistant}, AddGenerationPrompt: new(false), ContinueFinalMessage: llama.ContinuationContent},
			prompt:  "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\nHi",
		},
		"success: Jinja parses the allowlist of function tools": {
			preset: presetTokenizeJinja,
			request: llama.ApplyTemplateRequest{
				Messages: []llama.Message{{Role: "user", Content: llama.TextContent("Weather?")}},
				Tools: []llama.Tool{{
					Type:         "function",
					DeferLoading: new(true),
					Function: &llama.FunctionDefinition{
						Name: "weather", Parameters: jsontext.Value(`{"type":"object"}`), DeferLoading: new(false),
					},
				}},
				ToolChoice: llama.ToolChoiceAllowedTools("required", "weather"),
			},
			prompt: "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := cmp.Or(tt.preset, presetStories260K).server(t).Client(t)
			res, err := c.ApplyTemplate(t.Context(), &tt.request)
			if err != nil {
				t.Fatal(err)
			}
			if diff := gocmp.Diff(tt.prompt, res.Prompt); diff != "" {
				t.Errorf("rendered prompt differs (-want +got):\n%s", diff)
			}
		})
	}
}

func TestTokenizeEndpointErrors(t *testing.T) {
	c, _ := presetStories260K.server(t).Client(t)
	tests := map[string]struct {
		call func(*testing.T) error
		want string
	}{
		"error: tokenize does not accept a prompt object": {
			call: func(t *testing.T) error {
				_, err := llama.Call[llama.TokenizeResponse](t.Context(), c, http.MethodPost, "/tokenize", jsontext.Value(`{"content":{}}`))
				return err
			},
			want: "type must be string, but is object",
		},
		"error: detokenize expects a token array": {
			call: func(t *testing.T) error {
				_, err := llama.Call[llama.DetokenizeResponse](t.Context(), c, http.MethodPost, "/detokenize", jsontext.Value(`{"tokens":"invalid"}`))
				return err
			},
			want: "type must be array, but is string",
		},
		"error: apply template requires messages": {
			call: func(t *testing.T) error {
				_, err := c.ApplyTemplate(t.Context(), &llama.ApplyTemplateRequest{})
				return err
			},
			want: "'messages' is required",
		},
		"error: apply template rejects continuation with a generation prompt": {
			call: func(t *testing.T) error {
				_, err := c.ApplyTemplate(t.Context(), &llama.ApplyTemplateRequest{
					Messages:             []llama.Message{{Role: "user", Content: llama.TextContent("Hello")}},
					AddGenerationPrompt:  new(true),
					ContinueFinalMessage: llama.ContinuationAuto,
				})
				return err
			},
			want: "Cannot set both add_generation_prompt and continue_final_message to true.",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			err := tt.call(t)
			apiErr, ok := errors.AsType[*llama.APIError](err)
			if !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest || !strings.Contains(apiErr.Message, tt.want) {
				t.Errorf("error = %v, want HTTP 400 invalid_request_error containing %q", err, tt.want)
			}
		})
	}
}
