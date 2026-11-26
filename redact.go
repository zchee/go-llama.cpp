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
	"fmt"
	"io"
	"net/url"
	"strconv"
	"strings"
)

// redactedText replaces the API key in error and log text.
const redactedText = "[REDACTED]"

// secret holds the API key behind a function, so that printing a Client by reflection (%#v of a copied value, or
// of a struct that contains one) shows a function address and never the key.
type secret func() string

// newSecret returns a secret holding s, or nil for an empty key.
func newSecret(s string) secret {
	if s == "" {
		return nil
	}
	return func() string { return s }
}

// value returns the key, or "" when none is set.
func (s secret) value() string {
	if s == nil {
		return ""
	}
	return s()
}

// redact replaces every spelling of the key that may appear in text: as is, URL-escaped, and escaped inside a
// quoted Go string such as a %q of a header.
func (s secret) redact(text string) string {
	key := s.value()
	if key == "" {
		return text
	}
	quoted := strconv.Quote(key)
	for _, form := range []string{key, url.QueryEscape(key), url.PathEscape(key), quoted[1 : len(quoted)-1]} {
		text = strings.ReplaceAll(text, form, redactedText)
	}
	return text
}

// redactError returns err when its text does not contain the key, and otherwise wraps it in a redactedError.
func (s secret) redactError(err error) error {
	if err == nil || s.value() == "" {
		return err
	}
	text := err.Error()
	clean := s.redact(text)
	if clean == text {
		return err
	}
	return &redactedError{err: err, text: clean}
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
