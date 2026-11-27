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
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

	"github.com/zchee/go-llama.cpp/internal/jsontest"
)

func TestTokenizeWire(t *testing.T) {
	tests := map[string]struct {
		request TokenizeRequest
		want    []TokenizeToken
	}{
		"success: ids": {
			request: TokenizeRequest{Content: TokenizeText("Hello world!")},
			want:    []TokenizeToken{TokenizeID(346), TokenizeID(306), TokenizeID(414), TokenizeID(263), TokenizeID(304), TokenizeID(341), TokenizeID(443)},
		},
		"success: text_pieces": {
			request: TokenizeRequest{Content: TokenizeParts(TextPart("Hello"), TokenPart(0), TextPart("!")), AddSpecial: new(true), ParseSpecial: new(false), WithPieces: new(true)},
			want: []TokenizeToken{
				TokenizeIDWithPiece(1, TokenPieceText("<s>")), TokenizeIDWithPiece(346, TokenPieceText(" He")),
				TokenizeIDWithPiece(306, TokenPieceText("ll")), TokenizeIDWithPiece(414, TokenPieceText("o")),
				TokenizeIDWithPiece(0, TokenPieceText("<unk>")), TokenizeIDWithPiece(410, TokenPieceText(" ")),
				TokenizeIDWithPiece(443, TokenPieceText("!")),
			},
		},
		"success: byte_pieces": {
			request: TokenizeRequest{Content: TokenizeText("á"), WithPieces: new(true)},
			want: []TokenizeToken{
				TokenizeIDWithPiece(410, TokenPieceText(" ")),
				TokenizeIDWithPiece(198, TokenPieceBytes([]byte{195})), TokenizeIDWithPiece(164, TokenPieceBytes([]byte{161})),
			},
		},
		"success: token_input": {
			request: TokenizeRequest{Content: TokenizeParts(TokenPart(198), TokenPart(164))},
			want:    []TokenizeToken{TokenizeID(198), TokenizeID(164)},
		},
		"success: text_input": {
			request: TokenizeRequest{Content: TokenizeParts(TextPart("Hello"), TextPart("world!"))},
			want:    []TokenizeToken{TokenizeID(346), TokenizeID(306), TokenizeID(414), TokenizeID(263), TokenizeID(304), TokenizeID(341), TokenizeID(443)},
		},
		"success: parsed_special": {
			request: TokenizeRequest{Content: TokenizeText("<s>")},
			want:    []TokenizeToken{TokenizeID(1)},
		},
		"success: literal_special": {
			request: TokenizeRequest{Content: TokenizeText("<s>"), ParseSpecial: new(false)},
			want:    []TokenizeToken{TokenizeID(410), TokenizeID(504), TokenizeID(419), TokenizeID(505)},
		},
		"success: empty": {
			request: TokenizeRequest{Content: TokenizeParts()},
			want:    []TokenizeToken{},
		},
		"success: absent": {want: []TokenizeToken{}},
	}
	jsontest.CheckFallbacks[TokenizeRequest](t)
	jsontest.CheckFallbacks[TokenizeResponse](t)
	jsontest.CheckFallbacks[tokenizeTokenObject](t)
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			fixture := "testdata/tokenize/" + name[len("success: "):]
			tokenizeCheckRequest(t, "/tokenize", fixture, tt.request)
			got := tokenizeFixture[TokenizeResponse](t, fixture+".resp.json")
			if diff := gocmp.Diff(tt.want, got.Tokens, gocmp.AllowUnexported(TokenPiece{})); diff != "" {
				t.Errorf("tokens differ (-want +got):\n%s", diff)
			}
			text, isText := tt.request.Content.Text()
			if tt.request.Content.IsZero() == (isText || tt.request.Content.Parts() != nil) {
				t.Errorf("content state is inconsistent: text = %q, is text = %t, parts = %v", text, isText, tt.request.Content.Parts())
			}
		})
	}
}

func TestTokenizeErrorWire(t *testing.T) {
	tests := map[string]struct {
		fixture string
		message string
	}{
		"error: tokenize content is an object":  {"tokenize", "[json.exception.type_error.302] type must be string, but is object"},
		"error: detokenize tokens is a string":  {"detokenize", "[json.exception.type_error.302] type must be array, but is string"},
		"error: apply template messages absent": {"apply-template", "'messages' is required"},
	}
	jsontest.CheckFallbacks[errorEnvelope](t)
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			data := jsontest.ReadFile(t, "testdata/"+tt.fixture+"/error.resp.json")
			tokenizeFixture[errorEnvelope](t, "testdata/"+tt.fixture+"/error.resp.json")
			got := decodeAPIError(http.StatusBadRequest, nil, data)
			want := &APIError{StatusCode: http.StatusBadRequest, Code: http.StatusBadRequest, Message: tt.message, Type: ErrorTypeInvalidRequest, Raw: data}
			if diff := gocmp.Diff(want, got); diff != "" {
				t.Errorf("API error differs (-want +got):\n%s", diff)
			}
		})
	}
}

func tokenizeFixture[T any](t *testing.T, file string) *T {
	t.Helper()
	return jsontest.CheckWire[T](t, jsontest.ReadFile(t, file))
}

func tokenizeCheckRequest[T any](t *testing.T, path, fixture string, request T) {
	t.Helper()
	file := fixture + ".req.json"
	tokenizeFixture[T](t, file)
	c, err := New("http://127.0.0.1:8080/api")
	if err != nil {
		t.Fatal(err)
	}
	req, err := c.NewRequest(t.Context(), http.MethodPost, path, &request, WithModelQuery("stories"))
	if err != nil {
		t.Fatal(err)
	}
	defer req.Body.Close()
	if got := req.Method + " " + req.URL.RequestURI(); got != "POST /api"+path+"?model=stories" {
		t.Errorf("request = %q, want POST /api%s?model=stories", got, path)
	}
	if got := req.Header.Get("Content-Type"); got != "application/json" {
		t.Errorf("content type = %q, want application/json", got)
	}
	got, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatal(err)
	}
	wantJSON := jsontext.Value(jsontest.ReadFile(t, file))
	gotJSON := jsontext.Value(got)
	if err := wantJSON.Canonicalize(); err != nil {
		t.Fatal(err)
	}
	if err := gotJSON.Canonicalize(); err != nil {
		t.Fatal(err)
	}
	if diff := gocmp.Diff(wantJSON.String(), gotJSON.String()); diff != "" {
		t.Errorf("request body differs (-want +got):\n%s", diff)
	}
}

func tokenizeFuzzRoundTrip[T any](t *testing.T, data []byte) {
	t.Helper()
	var before, after T
	if err := json.Unmarshal(data, &before, jsonOptions); err != nil {
		return
	}
	encoded, err := json.Marshal(before, jsonOptions)
	if err != nil {
		t.Fatalf("encoding decoded %T: %v (input %q)", before, err, data)
	}
	if err := json.Unmarshal(encoded, &after, jsonOptions); err != nil {
		t.Fatalf("decoding re-encoded %T: %v (input %q, encoded %q)", before, err, data, encoded)
	}
	equalRawJSON := gocmp.FilterPath(func(p gocmp.Path) bool {
		return p.Last().Type() == reflect.TypeFor[jsontext.Value]()
	}, gocmp.Comparer(func(x, y jsontext.Value) bool {
		x, y = x.Clone(), y.Clone()
		opts := json.JoinOptions(jsontext.CanonicalizeRawInts(false), jsontext.CanonicalizeRawFloats(false))
		return x.Canonicalize(opts) == nil && y.Canonicalize(opts) == nil && x.String() == y.String()
	}))
	if diff := gocmp.Diff(before, after, gocmp.AllowUnexported(TokenizeContent{}, PromptPart{}, TokenPiece{}), equalRawJSON); diff != "" {
		t.Fatalf("%T changed after round trip (-before +after):\n%s\ninput %q, re-encoded %q", before, diff, data, encoded)
	}
}

func FuzzTokenizeContent(f *testing.F) {
	for _, stem := range []string{"ids", "text_pieces", "token_input", "text_input", "empty", "error"} {
		var request struct {
			Content jsontext.Value `json:"content"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/tokenize/"+stem+".req.json"), &request); err != nil {
			f.Fatal(err)
		}
		f.Add([]byte(request.Content))
	}
	f.Fuzz(tokenizeFuzzRoundTrip[TokenizeContent])
}

func FuzzTokenizeToken(f *testing.F) {
	for _, stem := range []string{"ids", "text_pieces", "byte_pieces"} {
		var response struct {
			Tokens []jsontext.Value `json:"tokens"`
		}
		if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/tokenize/"+stem+".resp.json"), &response); err != nil {
			f.Fatal(err)
		}
		for _, token := range response.Tokens {
			f.Add([]byte(token))
		}
	}
	f.Fuzz(tokenizeFuzzRoundTrip[TokenizeToken])
}

func FuzzTokenPiece(f *testing.F) {
	var response struct {
		Tokens []struct {
			Piece jsontext.Value `json:"piece"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal(jsontest.ReadFile(f, "testdata/tokenize/byte_pieces.resp.json"), &response); err != nil {
		f.Fatal(err)
	}
	for _, token := range response.Tokens {
		f.Add([]byte(token.Piece))
	}
	f.Fuzz(tokenizeFuzzRoundTrip[TokenPiece])
}

func ExampleClient_Tokenize() {
	c, err := New("http://127.0.0.1:8080")
	if err != nil {
		panic(err)
	}
	res, err := c.Tokenize(context.Background(), &TokenizeRequest{Content: TokenizeText("Hello")})
	if err != nil {
		panic(err)
	}
	for _, token := range res.Tokens {
		fmt.Println(token.ID)
	}
}
