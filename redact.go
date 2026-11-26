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
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// redactedText replaces the API key in error and log text.
const redactedText = "[REDACTED]"

// secret holds the API key behind a function, so that printing a Client by reflection (%#v of a copied value, or
// of a struct that contains one) shows a function address and never the key.
type secret func() (string, *regexp.Regexp)

// newSecret returns a secret holding s, or nil for an empty key.
func newSecret(s string) secret {
	if s == "" {
		return nil
	}
	// Header dumps contain the literal key or its Go/JSON string contents. In URLs any UTF-8
	// character may be literal or percent-encoded, and a query space may also be '+'. Only the
	// hex digits in percent escapes are case-insensitive. This does not decode nested encodings.
	var pattern strings.Builder
	for _, r := range s {
		lit := string(r)
		pattern.WriteString("(?:")
		pattern.WriteString(regexp.QuoteMeta(lit))
		if r == ' ' {
			pattern.WriteString(`|\+`)
		}
		pattern.WriteString("|(?i:")
		for i := range len(lit) {
			fmt.Fprintf(&pattern, "%%%02X", lit[i])
		}
		pattern.WriteString("))")
	}
	plain, _ := json.Marshal(s)
	q, _ := json.Marshal(s, jsontext.EscapeForHTML(true), jsontext.EscapeForJS(true))
	for _, quoted := range []string{strconv.Quote(s), strconv.QuoteToASCII(s), string(plain), string(q)} {
		if len(quoted) >= 2 && quoted[1:len(quoted)-1] != s {
			pattern.WriteByte('|')
			pattern.WriteString(regexp.QuoteMeta(quoted[1 : len(quoted)-1]))
		}
	}
	re, err := regexp.Compile(pattern.String())
	if err != nil || !utf8.ValidString(s) {
		return func() (string, *regexp.Regexp) { return s, nil }
	}
	re.Longest()
	return func() (string, *regexp.Regexp) { return s, re }
}

// value returns the key, or "" when none is set.
func (s secret) value() string {
	if s == nil {
		return ""
	}
	key, _ := s()
	return key
}

func (s secret) redact(text string) string {
	if s == nil || text == "" {
		return text
	}
	_, re := s()
	if re == nil {
		return redactedText
	}
	return re.ReplaceAllString(text, redactedText)
}

// redactError wraps keyed errors even when only their fields, not their message, contain the key.
func (s secret) redactError(err error) error {
	if err == nil || s == nil {
		return err
	}
	return &redactedError{err: err, text: s.redact(err.Error())}
}

// redactedError is an error whose text had the API key removed. Unwrap keeps the original chain available to
// errors.Is and errors.As.
type redactedError struct {
	err  error
	text string
}

func (e *redactedError) Error() string { return e.text }

func (e *redactedError) Unwrap() error { return e.err }

// Format implements fmt.Formatter so that %+v and %#v print the redacted text instead of the wrapped error's
// fields.
func (e *redactedError) Format(f fmt.State, verb rune) {
	if verb == 'q' {
		_, _ = io.WriteString(f, strconv.Quote(e.text))
		return
	}
	_, _ = io.WriteString(f, e.text)
}
