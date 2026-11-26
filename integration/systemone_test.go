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
	"errors"
	"math"
	"net/http"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
	"github.com/zchee/go-llama.cpp/openai"
	"github.com/zchee/go-llama.cpp/systemone"
)

var presetSystemone = &serverPreset{
	name: "systemone",
	options: func(t testing.TB) harness.Options {
		return harness.Options{
			Model: harness.Model(t, "tinyopenjev"),
			Args: []string{
				"--alias", "tinyopenjev", "--ctx-size", "4096", "--batch-size", "512",
				"--parallel", "1", "--seed", "42", "--mmproj", harness.Model(t, "tinyopenjev-mmproj"),
			},
		}
	},
}

func TestSystemoneEvaluation(t *testing.T) {
	const imageURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZ7kAAAAASUVORK5CYII="
	state := systemone.Text("A parcel has arrived.")
	q := map[string]systemone.Question{
		"route": systemone.Choice(systemone.Text("Which team?"), systemone.ChoiceCriteria{
			"billing": systemone.Null(), "shipping": systemone.Text("deliveries"),
		}),
		"urgency":   systemone.Score(systemone.Text("How urgent?"), systemone.ScoreCriteria{systemone.Text("low"), systemone.Text("high")}),
		"delivered": systemone.Noul(systemone.Text("Has it arrived?"), nil),
	}
	c, _ := presetSystemone.server(t).Client(t)
	client := systemone.NewClient(c)
	base, err := client.Evaluate(t.Context(), &systemone.Request{State: state, Questions: q})
	if err != nil {
		t.Fatal(err)
	}
	choice, ok := base.Answers["route"].Value.(systemone.ChoiceAnswer)
	if !ok || len(choice.Probabilities) != 2 || math.Abs(choice.Probabilities["billing"]+choice.Probabilities["shipping"]-1) > 1e-5 || choice.Probabilities[choice.Choice] < 0.5 {
		t.Fatalf("invalid choice: %+v", base.Answers["route"])
	}
	score, ok := base.Answers["urgency"].Value.(systemone.ScoreAnswer)
	if !ok || len(score.Legend) != 2 || math.Abs(score.Score-score.Probabilities["1"]) > 1e-5 {
		t.Fatalf("invalid expected level: %+v", base.Answers["urgency"])
	}
	messages, err := systemone.NewPayload([]openai.Message{{Role: "user", Content: openai.PartsContent(openai.ImageURLPart(imageURL), openai.TextPart("A parcel has arrived."))}})
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		preset *serverPreset
		req    systemone.Request
		status int
		kind   llama.ErrorType
	}{
		"success: files":               {preset: presetSystemone, req: systemone.Request{State: state, Questions: q, Files: []string{imageURL}}},
		"success: images alias":        {preset: presetSystemone, req: systemone.Request{State: state, Questions: q, Images: []string{imageURL}}},
		"success: chat image":          {preset: presetSystemone, req: systemone.Request{State: messages, Questions: q}},
		"error: null state on openjev": {preset: presetSystemone, req: systemone.Request{State: systemone.Null(), Questions: q}, status: http.StatusBadRequest, kind: llama.ErrorTypeInvalidRequest},
		"error: text model":            {preset: presetStories260K, req: systemone.Request{State: state, Questions: q}, status: http.StatusNotImplemented, kind: llama.ErrorTypeNotSupported},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			c, _ := tt.preset.server(t).Client(t)
			r, err := systemone.NewClient(c).Evaluate(t.Context(), &tt.req)
			if tt.status != 0 {
				e, ok := errors.AsType[*llama.APIError](err)
				if !ok || e.StatusCode != tt.status || e.Type != tt.kind {
					t.Fatalf("Evaluate error = %v, want HTTP %d %s", err, tt.status, tt.kind)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if r.Usage.InputTokens <= base.Usage.InputTokens {
				t.Errorf("image input used %d tokens, want more than text-only %d", r.Usage.InputTokens, base.Usage.InputTokens)
			}
		})
	}
}
