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

package llama

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"iter"
	"net/http"
	"slices"
)

// ModelList is the response from both /models and /v1/models.
type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
	// Models contains the Ollama-compatible listing in single-model mode only.
	Models []ModelDescription        `json:"models,omitzero"`
	Extra  map[string]jsontext.Value `json:",embed"`
}

// Model describes a model known to the server. Status and Source are router-only.
type Model struct {
	ID                 string                    `json:"id"`
	Aliases            []string                  `json:"aliases"`
	Tags               []string                  `json:"tags"`
	Object             string                    `json:"object"`
	OwnedBy            string                    `json:"owned_by"`
	Created            int64                     `json:"created"`
	Architecture       ModelArchitecture         `json:"architecture"`
	Meta               *ModelMetadata            `json:"meta,omitzero"`
	Status             *ModelStatus              `json:"status,omitzero"`
	Source             string                    `json:"source,omitzero"`
	CanRemove          *bool                     `json:"can_remove,omitzero"`
	ContextLength      *int                      `json:"context_length,omitzero"`
	Memory             *ModelMemory              `json:"memory,omitzero"`
	Sidecars           []string                  `json:"sidecars,omitzero"`
	IncompleteSidecars []string                  `json:"incomplete_sidecars,omitzero"`
	Extra              map[string]jsontext.Value `json:",embed"`
}

// ModelArchitecture lists the modalities accepted and produced by a model.
type ModelArchitecture struct {
	InputModalities  []string                  `json:"input_modalities"`
	OutputModalities []string                  `json:"output_modalities"`
	Extra            map[string]jsontext.Value `json:",embed"`
}

// ModelMetadata reports the loaded model's dimensions, vocabulary, and file type.
type ModelMetadata struct {
	VocabType int                       `json:"vocab_type"`
	NVocab    int                       `json:"n_vocab"`
	NCtx      int                       `json:"n_ctx"`
	NCtxTrain int                       `json:"n_ctx_train"`
	NEmbd     int                       `json:"n_embd"`
	NParams   uint64                    `json:"n_params"`
	Size      uint64                    `json:"size"`
	FType     string                    `json:"ftype"`
	Extra     map[string]jsontext.Value `json:",embed"`
}

// ModelDescription is the Ollama-compatible model entry from a single-model server.
type ModelDescription struct {
	Name         string                    `json:"name"`
	Model        string                    `json:"model"`
	ModifiedAt   string                    `json:"modified_at"`
	Size         string                    `json:"size"`
	Digest       string                    `json:"digest"`
	Type         string                    `json:"type"`
	Description  string                    `json:"description"`
	Tags         []string                  `json:"tags"`
	Capabilities []string                  `json:"capabilities"`
	Parameters   string                    `json:"parameters"`
	Details      ModelDetails              `json:"details"`
	Extra        map[string]jsontext.Value `json:",embed"`
}

// ModelDetails contains the Ollama-compatible description of the model file.
type ModelDetails struct {
	ParentModel       string                    `json:"parent_model"`
	Format            string                    `json:"format"`
	Family            string                    `json:"family"`
	Families          []string                  `json:"families"`
	ParameterSize     string                    `json:"parameter_size"`
	QuantizationLevel string                    `json:"quantization_level"`
	Extra             map[string]jsontext.Value `json:",embed"`
}

// ModelStatus describes the router child process; a failed exit is reported as unloaded.
type ModelStatus struct {
	Value    ModelState                `json:"value"`
	Args     []string                  `json:"args"`
	Preset   *string                   `json:"preset,omitzero"`
	Failed   *bool                     `json:"failed,omitzero"`
	ExitCode *int                      `json:"exit_code,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// ModelState identifies a router model lifecycle state.
type ModelState string

// Model lifecycle states reported by the router.
const (
	ModelDownloading ModelState = "downloading"
	ModelIncomplete  ModelState = "incomplete"
	ModelDownloaded  ModelState = "downloaded"
	ModelUnloaded    ModelState = "unloaded"
	ModelLoading     ModelState = "loading"
	ModelLoaded      ModelState = "loaded"
	ModelSleeping    ModelState = "sleeping"
)

// ModelMemory describes the router's memory measurements; unknown measurements are null.
type ModelMemory struct {
	Status    string                    `json:"status"`
	Min       *ModelMemoryUsage         `json:"min"`
	Max       *ModelMemoryUsage         `json:"max"`
	Effective *ModelMemoryUsage         `json:"effective"`
	Extra     map[string]jsontext.Value `json:",embed"`
}

// ModelMemoryUsage describes allocations at the specified context size.
type ModelMemoryUsage struct {
	NCtx    uint32                    `json:"n_ctx"`
	Devices []ModelMemoryDevice       `json:"devices"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// ModelMemoryDevice reports byte counts; only accelerators report Total and Free.
type ModelMemoryDevice struct {
	Kind    string                    `json:"kind"`
	Name    string                    `json:"name"`
	Model   uint64                    `json:"model"`
	Context uint64                    `json:"context"`
	Compute uint64                    `json:"compute"`
	Total   *uint64                   `json:"total,omitzero"`
	Free    *uint64                   `json:"free,omitzero"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// ModelRequest names the model to load or unload.
type ModelRequest struct {
	Model string                    `json:"model"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// DownloadModelRequest selects a Hugging Face model and optional sidecars.
type DownloadModelRequest struct {
	Model string `json:"model"`
	// A nil slice uses the server default; an empty slice downloads no sidecars.
	Sidecars []string                  `json:"sidecars,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// ModelActionResponse reports whether the server accepted the requested operation.
type ModelActionResponse struct {
	Success bool                      `json:"success"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// WithReload asks the router to rescan its model sources before listing them.
func WithReload() RequestOption { return WithQuery("reload", "1") }

// WithSidecar restricts a cache deletion to a single sidecar kind.
func WithSidecar(kind string) RequestOption { return WithQuery("sidecar", kind) }

// ListModels returns the models known to the server.
func (c *Client) ListModels(ctx context.Context, opts ...RequestOption) (*ModelList, error) {
	return Call[*ModelList](ctx, c, http.MethodGet, "/models", nil, opts...)
}

// LoadModel asks the router to start loading the named model.
func (c *Client) LoadModel(ctx context.Context, req *ModelRequest, opts ...RequestOption) (*ModelActionResponse, error) {
	return Call[*ModelActionResponse](ctx, c, http.MethodPost, "/models/load", req, opts...)
}

// UnloadModel asks the router to unload a running model or pause a download.
func (c *Client) UnloadModel(ctx context.Context, req *ModelRequest, opts ...RequestOption) (*ModelActionResponse, error) {
	return Call[*ModelActionResponse](ctx, c, http.MethodPost, "/models/unload", req, opts...)
}

// DownloadModel starts a background download into the router's cache. Observe ModelEvents for completion.
func (c *Client) DownloadModel(ctx context.Context, req *DownloadModelRequest, opts ...RequestOption) (*ModelActionResponse, error) {
	return Call[*ModelActionResponse](ctx, c, http.MethodPost, "/models", req, opts...)
}

// DeleteModel removes the named model from the router's cache, or one sidecar with WithSidecar.
// It rejects an empty model name, or an empty or repeated sidecar selector rather than risk deleting the whole model.
func (c *Client) DeleteModel(ctx context.Context, model string, opts ...RequestOption) (*ModelActionResponse, error) {
	if model == "" {
		return nil, errors.New("llama: model deletion requires a non-empty model name")
	}
	if kinds, ok := newRequestConfig(opts).query["sidecar"]; ok && (len(kinds) != 1 || kinds[0] == "") {
		return nil, errors.New("llama: model deletion requires a single non-empty sidecar when one is specified")
	}
	return Call[*ModelActionResponse](ctx, c, http.MethodDelete, "/models", nil, append(slices.Clip(opts), WithModelQuery(model))...)
}

// ModelEvents subscribes to model changes. Break the loop or cancel ctx to close the stream. The sequence also ends
// without an error when the server closes the stream (for example on shutdown), because this event stream has no
// terminating event, so a caller cannot tell a server shutdown from a normal end.
func (c *Client) ModelEvents(ctx context.Context, opts ...RequestOption) iter.Seq2[*ModelEvent, error] {
	return Stream[ModelEvent](ctx, c, http.MethodGet, "/models/sse", nil, opts...)
}
