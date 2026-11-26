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
	"context"
	"errors"
	"fmt"
	"net/url"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/openai"
)

var _ = registerContracts(map[string]contractCase{
	"GET /models": {server: modelsPreset("listing", false), query: url.Values{"reload": {"1"}}, call: func(t *testing.T, c *llama.Client) error {
		list, err := c.ListModels(t.Context(), llama.WithReload())
		if err == nil && (len(list.Data) != 1 || list.Data[0].ID != modelsLocalID || list.Data[0].Status == nil || list.Data[0].Status.Value != llama.ModelUnloaded) {
			return fmt.Errorf("unexpected router list: %+v", list)
		}
		return err
	}},
	"GET /v1/models": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		list, err := openai.NewClient(c).Models(t.Context())
		if err == nil && (len(list.Data) != 1 || list.Data[0].Meta == nil || len(list.Models) != 1) {
			return fmt.Errorf("missing single-model metadata: %+v", list)
		}
		return err
	}},
	"POST /models/load": {server: modelsPreset("load", false), call: func(t *testing.T, c *llama.Client) error {
		res, err := c.LoadModel(t.Context(), &llama.ModelRequest{Model: modelsLocalID})
		if err == nil && !res.Success {
			return errors.New("load was not accepted")
		}
		return err
	}},
	"POST /models/unload": {server: modelsPreset("unload", false), call: func(t *testing.T, c *llama.Client) error {
		req := &llama.ModelRequest{Model: modelsLocalID}
		if _, err := c.LoadModel(t.Context(), req); err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
		defer cancel()
		if _, err := modelsWait(ctx, c, modelsLocalID, llama.ModelLoaded); err != nil {
			return err
		}
		res, err := c.UnloadModel(ctx, req)
		if err == nil && !res.Success {
			return errors.New("unload was not accepted")
		}
		return err
	}},
	"GET /models/sse": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error {
		ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
		defer cancel()
		for ev, err := range c.ModelEvents(ctx) {
			if err != nil {
				return err
			}
			status, ok := ev.Data.(*llama.ModelStatusChange)
			if !ok || status.Status != llama.ModelLoaded {
				return fmt.Errorf("unexpected replayed event: %+v", ev)
			}
			return nil
		}
		return errors.New("no replayed event")
	}},
	"POST /models": {server: modelsDownloadPreset, call: func(t *testing.T, c *llama.Client) error {
		modelsDownloadAndCheck(t, c, modelsDownloadPreset.server(t))
		return nil
	}},
	"DELETE /models": {server: modelsDeletePreset, query: url.Values{"model": {modelsHubID}}, call: func(t *testing.T, c *llama.Client) error {
		file := modelsDownloadAndCheck(t, c, modelsDeletePreset.server(t))
		sidecar, err := c.DeleteModel(t.Context(), modelsHubID, llama.WithSidecar("mmproj"))
		if err != nil {
			return err
		}
		if sidecar.Success {
			return errors.New("removed a sidecar that was not downloaded")
		}
		res, err := c.DeleteModel(t.Context(), modelsHubID)
		if err != nil {
			return err
		}
		if !res.Success {
			return errors.New("model not removed")
		}
		return modelsCheckDeleted(file)
	}},
})
