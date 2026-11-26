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
	numState, err := systemone.NewPayload(42)
	if err != nil {
		t.Fatal(err)
	}
	boolState, err := systemone.NewPayload(true)
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
		"success: numeric state":       {preset: presetSystemone, req: systemone.Request{State: numState, Questions: map[string]systemone.Question{"even": systemone.Noul(systemone.Text("Is this number even?"), nil)}}},
		"success: boolean state":       {preset: presetSystemone, req: systemone.Request{State: boolState, Questions: map[string]systemone.Question{"positive": systemone.Noul(systemone.Text("Is this value true?"), nil)}}},
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
			if _, ok := tt.req.Questions["route"]; ok && r.Usage.InputTokens <= base.Usage.InputTokens {
				t.Errorf("image input used %d tokens, want more than text-only %d", r.Usage.InputTokens, base.Usage.InputTokens)
			}
			if r.Usage.InputTokens <= 0 {
				t.Errorf("input tokens = %d, want positive", r.Usage.InputTokens)
			}
		})
	}
}

func TestSystemoneRouter(t *testing.T) {
	dir := modelsDirWith(t, "tinyopenjev")
	srv := harness.StartT(t, harness.Options{ModelsDir: dir, Args: []string{"--ctx-size", "4096", "--batch-size", "512", "--parallel", "1", "--seed", "42"}})
	c, _ := srv.Client(t)
	client := systemone.NewClient(c)
	q := map[string]systemone.Question{"delivered": systemone.Noul(systemone.Text("Has it arrived?"), nil)}

	if _, err := client.Evaluate(t.Context(), &systemone.Request{State: systemone.Text("A parcel has arrived."), Questions: q}); err == nil {
		t.Fatal("unrouted POST /v1/systemone succeeded, want HTTP 400")
	} else if apiErr, ok := errors.AsType[*llama.APIError](err); !ok || apiErr.StatusCode != http.StatusBadRequest || apiErr.Type != llama.ErrorTypeInvalidRequest || apiErr.Message != "model name is missing from the request" {
		t.Fatalf("unrouted POST /v1/systemone error = %v, want HTTP 400 model name is missing", err)
	}

	res, err := client.Evaluate(t.Context(), &systemone.Request{Model: "tinyopenjev", State: systemone.Text("A parcel has arrived."), Questions: q})
	if err != nil {
		t.Fatalf("routed POST /v1/systemone failed: %v", err)
	}
	ans, ok := res.Answers["delivered"].Value.(systemone.NoulAnswer)
	if !ok || ans.Noul < 0 || ans.Noul > 1 || res.Usage.InputTokens <= 0 || res.Usage.OutputTokens != 0 {
		t.Fatalf("unexpected routed response: %+v", res)
	}
}
