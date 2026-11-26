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
	"encoding/json/jsontext"
	json "encoding/json/v2"
)

// ModelEvent identifies a model change. Data is nil for notifications without a payload.
type ModelEvent struct {
	Model string                    `json:"model"`
	Event string                    `json:"event"`
	Data  ModelEventData            `json:"data,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelEventData is a status change, memory measurement, download update or unrecognized payload.
type ModelEventData interface{ modelsEventData() }

// ModelStatusChange is the payload of a model_status or status_change event.
type ModelStatusChange struct {
	Status   ModelState                `json:"status"`
	ExitCode *int                      `json:"exit_code,omitzero"`
	Info     *Model                    `json:"info,omitzero"`
	Memory   *ModelMemoryUsage         `json:"memory,omitzero"`
	Progress *ModelLoadProgress        `json:"progress,omitzero"`
	Extra    map[string]jsontext.Value `json:",embed"`
}

// ModelLoadProgress reports the current stage of loading a model.
type ModelLoadProgress struct {
	Stages  []string `json:"stages,omitzero"`
	Current *string  `json:"current,omitzero"`
	Value   *float64 `json:"value,omitzero"`
	// Stage alone is sent when the projector starts loading.
	Stage *string                   `json:"stage,omitzero"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelDownloadProgress maps each file URL to its byte counts.
type ModelDownloadProgress struct {
	Progress map[string]ModelFileProgress `json:"progress"`
	Extra    map[string]jsontext.Value    `json:",embed"`
}

// ModelFileProgress reports how many bytes a file download has received.
type ModelFileProgress struct {
	Done  uint64                    `json:"done"`
	Total uint64                    `json:"total"`
	Extra map[string]jsontext.Value `json:",embed"`
}

// ModelDownloadFailure identifies a download stopped by a client.
type ModelDownloadFailure struct {
	Stopped bool                      `json:"stopped"`
	Extra   map[string]jsontext.Value `json:",embed"`
}

// ModelRawEventData preserves an unrecognized event payload verbatim.
type ModelRawEventData struct{ jsontext.Value }

func (*ModelStatusChange) modelsEventData()     {}
func (*ModelMemory) modelsEventData()           {}
func (*ModelDownloadProgress) modelsEventData() {}
func (*ModelDownloadFailure) modelsEventData()  {}
func (*ModelRawEventData) modelsEventData()     {}

// UnmarshalJSONFrom decodes the payload by event name, independent of member order.
func (e *ModelEvent) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var wire struct {
		Model string                    `json:"model"`
		Event string                    `json:"event"`
		Data  jsontext.Value            `json:"data"`
		Extra map[string]jsontext.Value `json:",embed"`
	}
	if err := json.UnmarshalDecode(dec, &wire); err != nil {
		return err
	}
	var data ModelEventData
	if wire.Data.Kind() == 'n' {
		data = &ModelRawEventData{wire.Data}
	} else if wire.Data != nil {
		switch wire.Event {
		case "status_change", "model_status":
			data = new(ModelStatusChange)
		case "model_memory":
			data = new(ModelMemory)
		case "download_progress":
			data = new(ModelDownloadProgress)
		case "download_failed":
			data = new(ModelDownloadFailure)
		default:
			data = &ModelRawEventData{wire.Data}
		}
		if _, raw := data.(*ModelRawEventData); !raw {
			if err := json.Unmarshal(wire.Data, data, dec.Options()); err != nil {
				return err
			}
		}
	}
	*e = ModelEvent{Model: wire.Model, Event: wire.Event, Data: data, Extra: wire.Extra}
	return nil
}
