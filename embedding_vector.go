// Copyright 2026 The go-llama.cpp Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package llama

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"math"
	"slices"
)

// EmbeddingVector is a numeric vector decoded from a float array or a base64 string.
// It marshals in the representation it was decoded from. Base64 decoding assumes a little-endian
// server. Request EmbeddingEncodingFloat when the server's byte order is unknown.
// Encoding after a write to the Values view is unspecified: a decoded float array may retain its original JSON.
type EmbeddingVector struct {
	values []float64
	format EmbeddingEncodingFormat
	// llama.cpp and Go can round the same float to different equivalent decimal spellings.
	floatJSON jsontext.Value
}

// EmbeddingVectorOf constructs a vector in the given wire format. An empty format uses a float array.
// Base64 vectors store values rounded to float32 precision.
func EmbeddingVectorOf(format EmbeddingEncodingFormat, values ...float64) EmbeddingVector {
	values = slices.Clone(values)
	if format == EmbeddingEncodingBase64 {
		for i, n := range values {
			values[i] = float64(float32(n))
		}
	}
	return EmbeddingVector{values: values, format: format}
}

// Len returns the number of values in the vector.
func (v EmbeddingVector) Len() int { return len(v.values) }

// Values returns a read-only view of the vector's numeric values.
func (v EmbeddingVector) Values() []float64 { return v.values }

// Format returns the vector's wire format.
func (v EmbeddingVector) Format() EmbeddingEncodingFormat {
	return cmp.Or(v.format, EmbeddingEncodingFloat)
}

// MarshalJSONTo implements json.MarshalerTo, keeping the vector's wire format.
func (v EmbeddingVector) MarshalJSONTo(enc *jsontext.Encoder) error {
	switch v.Format() {
	case EmbeddingEncodingFloat:
		if v.floatJSON != nil {
			return enc.WriteValue(v.floatJSON)
		}
		return json.MarshalEncode(enc, v.values)
	case EmbeddingEncodingBase64:
		data := make([]byte, 0, len(v.values)*4)
		for _, n := range v.values {
			f := float32(n)
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				return errors.New("llama: embedding value is not a finite float32")
			}
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(f))
		}
		return json.MarshalEncode(enc, data)
	default:
		return fmt.Errorf("llama: invalid embedding encoding format %q", v.format)
	}
}

// UnmarshalJSONFrom implements json.UnmarshalerFrom for float arrays and base64 strings.
func (v *EmbeddingVector) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	switch dec.PeekKind() {
	case '[':
		raw, err := dec.ReadValue()
		if err != nil {
			return err
		}
		if bytes.Contains(raw, []byte("null")) {
			return errors.New("llama: embedding vector contains a null value")
		}
		var nums []float64
		if err := json.Unmarshal(raw, &nums); err != nil {
			return err
		}
		*v = EmbeddingVector{values: nums, format: EmbeddingEncodingFloat, floatJSON: raw.Clone()}
		return nil
	case '"':
		var data []byte
		if err := json.UnmarshalDecode(dec, &data); err != nil {
			return err
		}
		if len(data)%4 != 0 {
			return errors.New("llama: base64 embedding does not contain whole float32 values")
		}
		nums := make([]float64, len(data)/4)
		for i := range nums {
			n := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:])))
			if math.IsNaN(n) || math.IsInf(n, 0) {
				return errors.New("llama: embedding value is not a finite float32")
			}
			nums[i] = n
		}
		*v = EmbeddingVector{values: nums, format: EmbeddingEncodingBase64}
		return nil
	default:
		return errors.New("llama: embedding vector must be a number array or a base64 string")
	}
}
