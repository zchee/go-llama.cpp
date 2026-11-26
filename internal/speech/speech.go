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

// Package speech encodes reference audio uploads and decodes speech response headers.
package speech

import (
	"bytes"
	"cmp"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"slices"
	"strconv"
	"strings"
)

// Format describes the encoding advertised by the speech response's Content-Type.
type Format struct {
	// MediaType is audio/wav or audio/pcm.
	MediaType string
	// SampleRate is the number of PCM sample frames per second.
	SampleRate int
	// Channels is the number of interleaved PCM channels, one when the header omits it.
	Channels int
	// Encoding is the PCM representation, such as float or signed-int.
	Encoding string
	// Bits is the number of bits per PCM sample.
	Bits int
}

// ResponseFormat checks the content type of a successful response and closes its body on error.
func ResponseFormat(resp *http.Response) (Format, error) {
	format, err := parseFormat(resp.Header.Get("Content-Type"))
	if err != nil {
		return Format{}, errors.Join(err, resp.Body.Close())
	}
	return format, nil
}

func parseFormat(contentType string) (Format, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return Format{}, err
	}
	f := Format{MediaType: mediaType}
	switch mediaType {
	case "audio/wav":
		return f, nil
	case "audio/pcm":
		f.Encoding = params["encoding"]
		if f.Encoding == "" {
			return Format{}, errors.New("llama: missing PCM encoding")
		}
		f.Channels = 1
		for key, dest := range map[string]*int{"rate": &f.SampleRate, "channels": &f.Channels, "bits": &f.Bits} {
			value, ok := params[key]
			if !ok && key == "channels" {
				continue
			}
			n, err := strconv.Atoi(value)
			if err != nil || n <= 0 {
				return Format{}, fmt.Errorf("llama: invalid PCM %s %q", key, value)
			}
			*dest = n
		}
		return f, nil
	default:
		return Format{}, fmt.Errorf("llama: unexpected speech media type %q", mediaType)
	}
}

// Multipart encodes request's JSON fields as form text and appends the uploaded reference.
func Multipart(request any, reference []byte, filename, contentType string) ([]byte, string, error) {
	data, err := json.Marshal(request, json.Deterministic(true))
	if err != nil {
		return nil, "", err
	}
	var fields map[string]jsontext.Value
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, "", err
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, k := range slices.Sorted(maps.Keys(fields)) {
		v := fields[k]
		text := string(v)
		if v.Kind() == '"' {
			if err := json.Unmarshal(v, &text); err != nil {
				return nil, "", err
			}
		}
		if strings.ContainsAny(k, "\r\n") {
			return nil, "", errors.New("llama: invalid speech form field name")
		}
		if err := w.WriteField(k, text); err != nil {
			return nil, "", err
		}
	}
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", mime.FormatMediaType("form-data", map[string]string{"name": "speaker_ref", "filename": cmp.Or(filename, "reference.wav")}))
	if strings.ContainsAny(contentType, "\r\n") {
		return nil, "", errors.New("llama: invalid speaker reference content type")
	}
	header.Set("Content-Type", cmp.Or(contentType, "application/octet-stream"))
	part, err := w.CreatePart(header)
	if err != nil {
		return nil, "", err
	}
	if _, err := part.Write(reference); err != nil {
		return nil, "", err
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return buf.Bytes(), w.FormDataContentType(), nil
}
