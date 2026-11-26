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
	// Object is the OpenAI-compatible discriminator "list".
	Object string `json:"object"`
	// Data contains one entry in single-model mode or the router's visible model entries.
	Data []Model `json:"data"`
	// Models contains the Ollama-compatible listing in single-model mode only.
	Models []ModelDescription `json:"models,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// Model describes a model known to the server. Status and Source are router-only.
type Model struct {
	// ID is the server's primary name for the model, usable in requests.
	ID string `json:"id"`
	// Aliases lists names the server recognizes when selecting this model.
	Aliases []string `json:"aliases"`
	// Tags lists labels configured for the model, with no entries when none were set.
	Tags []string `json:"tags"`
	// Object is the OpenAI-compatible discriminator "model".
	Object string `json:"object"`
	// OwnedBy is "llamacpp" for OpenAI compatibility, not the upstream model publisher.
	OwnedBy string `json:"owned_by"`
	// Created is when the server built the model info, in Unix seconds, not when the model was trained.
	Created int64 `json:"created"`
	// Architecture describes the supported modalities, inferred offline in the router until a child reports them.
	Architecture ModelArchitecture `json:"architecture"`
	// Meta reports dimensions from the loaded model and can be absent before a child reports them.
	Meta *ModelMetadata `json:"meta,omitzero"`
	// Status reports the router's model lifecycle state and is absent in single-model mode.
	Status *ModelStatus `json:"status,omitzero"`
	// Source is "cache", "models_dir", or "preset" in router mode and is absent in single-model mode.
	Source string `json:"source,omitzero"`
	// CanRemove reports whether the model comes from the router cache and is absent in single-model mode.
	CanRemove *bool `json:"can_remove,omitzero"`
	// ContextLength is the training context in tokens, omitted when the router does not know it.
	ContextLength *int `json:"context_length,omitzero"`
	// Memory holds the router's memory measurements when memory-based unloading is enabled.
	Memory *ModelMemory `json:"memory,omitzero"`
	// Sidecars lists the companion file kinds on disk or requested by an active or incomplete download.
	Sidecars []string `json:"sidecars,omitzero"`
	// IncompleteSidecars lists missing kinds with partial downloads and is omitted when there are none.
	IncompleteSidecars []string `json:"incomplete_sidecars,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelArchitecture lists the modalities accepted and produced by a model.
type ModelArchitecture struct {
	// InputModalities lists "text" and any media the model accepts, such as "image", "audio", or "video".
	InputModalities []string `json:"input_modalities"`
	// OutputModalities lists the outputs the model can generate, such as "text", "audio", or "decisions".
	OutputModalities []string `json:"output_modalities"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelMetadata reports the loaded model's dimensions, vocabulary, and file type.
type ModelMetadata struct {
	// VocabType is the numeric llama.cpp vocabulary type for the model's tokenizer.
	VocabType int `json:"vocab_type"`
	// NVocab counts entries in the model vocabulary.
	NVocab int `json:"n_vocab"`
	// NCtx is the runtime context size per slot, in tokens.
	NCtx int `json:"n_ctx"`
	// NCtxTrain is the model's training context size, in tokens.
	NCtxTrain int `json:"n_ctx_train"`
	// NEmbd is the width of one input embedding vector, in elements.
	NEmbd int `json:"n_embd"`
	// NParams counts the model's parameters.
	NParams uint64 `json:"n_params"`
	// Size is the sum of the model tensor sizes, in bytes.
	Size uint64 `json:"size"`
	// FType is the llama.cpp display name for the model's weight format or quantization.
	FType string `json:"ftype"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelDescription is the Ollama-compatible model entry from a single-model server.
type ModelDescription struct {
	// Name is the same server-assigned model name reported as Model.
	Name string `json:"name"`
	// Model is the primary name also reported as the ID of the corresponding ModelList.Data entry.
	Model string `json:"model"`
	// ModifiedAt is a compatibility placeholder returned as an empty string, not a file timestamp.
	ModifiedAt string `json:"modified_at"`
	// Size is an empty compatibility placeholder; see ModelMetadata.Size for the actual weight bytes.
	Size string `json:"size"`
	// Digest is an empty compatibility placeholder because the server does not track model file hashes.
	Digest string `json:"digest"`
	// Type is the fixed value "model".
	Type string `json:"type"`
	// Description is a compatibility placeholder returned as an empty string instead of model card text.
	Description string `json:"description"`
	// Tags contains one empty string for compatibility; see Model.Tags for configured labels.
	Tags []string `json:"tags"`
	// Capabilities lists "completion" and adds "multimodal" when a multimodal projector is loaded.
	Capabilities []string `json:"capabilities"`
	// Parameters is a compatibility placeholder returned as an empty string, not the active settings.
	Parameters string `json:"parameters"`
	// Details provides the GGUF format and compatibility placeholders for the other Ollama detail fields.
	Details ModelDetails `json:"details"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelDetails contains the Ollama-compatible description of the model file.
type ModelDetails struct {
	// ParentModel is a compatibility placeholder returned as an empty string, not a base model name.
	ParentModel string `json:"parent_model"`
	// Format is the fixed value "gguf".
	Format string `json:"format"`
	// Family is a compatibility placeholder returned as an empty string, not an architecture label.
	Family string `json:"family"`
	// Families is a one-element compatibility placeholder containing an empty string, not architecture labels.
	Families []string `json:"families"`
	// ParameterSize is an empty compatibility placeholder; see ModelMetadata.NParams for the count.
	ParameterSize string `json:"parameter_size"`
	// QuantizationLevel is an empty compatibility placeholder; ModelMetadata.FType describes the weight format.
	QuantizationLevel string `json:"quantization_level"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelStatus describes the router child process; a failed exit is reported as unloaded.
type ModelStatus struct {
	// Value is the current lifecycle state, including pending downloads and idle models.
	Value ModelState `json:"value"`
	// Args lists the arguments configured for the child process, even when the model is unloaded.
	Args []string `json:"args"`
	// Preset holds the router's INI configuration with reserved options removed, if a named preset exists.
	Preset *string `json:"preset,omitzero"`
	// Failed is present and true only when the child has exited unsuccessfully.
	Failed *bool `json:"failed,omitzero"`
	// ExitCode is the child process exit status, included only when Failed is true.
	ExitCode *int `json:"exit_code,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
	// Status is the measurement state: "unmeasured", "measuring", "measured", or "failed".
	Status string `json:"status"`
	// Min estimates usage at the smallest allowed context and is null if not measured.
	Min *ModelMemoryUsage `json:"min"`
	// Max estimates usage at the largest context that fit during measurement and is null if not measured.
	Max *ModelMemoryUsage `json:"max"`
	// Effective reports the last load's actual allocations, retained during sleep and cleared on unload.
	Effective *ModelMemoryUsage `json:"effective"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelMemoryUsage describes allocations at the specified context size.
type ModelMemoryUsage struct {
	// NCtx is the total context capacity in tokens across all slots for these allocations.
	NCtx uint32 `json:"n_ctx"`
	// Devices breaks down usage by accelerator, host memory, and any separately reported buffer types.
	Devices []ModelMemoryDevice `json:"devices"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelMemoryDevice reports byte counts; only accelerators report Total and Free.
type ModelMemoryDevice struct {
	// Kind classifies the row as "device", "host", or "buffer_type".
	Kind string `json:"kind"`
	// Name identifies the reporting device, host, or buffer type.
	Name string `json:"name"`
	// Model is the number of bytes used by model weights in this row.
	Model uint64 `json:"model"`
	// Context is the number of bytes used by the model context, including the KV cache.
	Context uint64 `json:"context"`
	// Compute is the number of bytes used by computation buffers.
	Compute uint64 `json:"compute"`
	// Total is the accelerator's memory capacity in bytes and is absent for other row kinds.
	Total *uint64 `json:"total,omitzero"`
	// Free is the accelerator's available memory in bytes when sampled and is absent for other row kinds.
	Free *uint64 `json:"free,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelRequest names the model to load or unload.
type ModelRequest struct {
	// Model is the required router model name or alias identifying the target of the operation.
	Model string `json:"model"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// DownloadModelRequest selects a Hugging Face model and optional sidecars.
type DownloadModelRequest struct {
	// Model is the required Hugging Face repository name, optionally followed by a colon and quantization tag.
	Model string `json:"model"`
	// Sidecars selects file kinds to download; nil defaults to ["mmproj"] and an empty slice selects none.
	Sidecars []string `json:"sidecars,omitzero"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelActionResponse reports whether the server accepted the requested operation.
type ModelActionResponse struct {
	// Success acknowledges the operation, not completion of a download or load;
	// false can mean the requested sidecar was absent.
	Success bool `json:"success"`
	// Extra holds members this type does not declare; they survive a round trip.
	Extra map[string]jsontext.Value `json:",embed"`
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
	if req == nil {
		return nil, errors.New("llama: nil load model request")
	}
	return Call[*ModelActionResponse](ctx, c, http.MethodPost, "/models/load", req, opts...)
}

// UnloadModel asks the router to unload a running model or pause a download.
func (c *Client) UnloadModel(ctx context.Context, req *ModelRequest, opts ...RequestOption) (*ModelActionResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil unload model request")
	}
	return Call[*ModelActionResponse](ctx, c, http.MethodPost, "/models/unload", req, opts...)
}

// DownloadModel starts a background download into the router's cache. Observe ModelEvents for completion.
func (c *Client) DownloadModel(ctx context.Context, req *DownloadModelRequest, opts ...RequestOption) (*ModelActionResponse, error) {
	if req == nil {
		return nil, errors.New("llama: nil download model request")
	}
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
