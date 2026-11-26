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

// Command record sends one request to a real llama-server and writes the request body and the response body as
// test fixtures, so that codec tests decode bytes the server actually produced.
//
// For a fixture prefix testdata/<endpoint>/<case> it writes the request body as sent, when there is one, as
// <case>.req.json (a -content-type that starts with application/json, the default) or <case>.req.bin (any
// other), and the response body as <case>.resp.json (application/json), <case>.resp.sse (text/event-stream),
// <case>.resp.txt (text/plain) or <case>.resp.bin (anything else).
//
// It either starts a server from a model of integration/models.yaml and stops it afterwards, or talks to a
// running one:
//
//	go run ./integration/cmd/record -model stories260K -server-args '--ctx-size 512' \
//		-method POST -path /completion -body '{"prompt":"Hello","n_predict":8}' \
//		-out testdata/completion/basic
//	go run ./integration/cmd/record -server http://127.0.0.1:8080 -method GET -path /health -out testdata/health/ok
//
// A request body is given inline with -body or read from a file with -body-file. -want-status (default 200)
// records an error response when set to its status; any other status fails without writing anything. The API key
// of a protected server is read from the environment variable named by -api-key-env and is never printed.
package main

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

type config struct {
	server      string
	socket      string
	model       string
	serverArgs  string
	apiKeyEnv   string
	method      string
	path        string
	body        string
	bodyFile    string
	contentType string
	out         string
	wantStatus  int
	timeout     time.Duration
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("record: ")
	var cfg config
	flag.StringVar(&cfg.server, "server", "", "base URL of a running server")
	flag.StringVar(&cfg.socket, "socket", "", "UNIX socket of a running server (with -server http://localhost)")
	flag.StringVar(&cfg.model, "model", "", "start a server with this model of integration/models.yaml")
	flag.StringVar(&cfg.serverArgs, "server-args", "", "extra flags of the started server, separated by spaces")
	flag.StringVar(&cfg.apiKeyEnv, "api-key-env", "", "environment variable holding the API key")
	flag.StringVar(&cfg.method, "method", http.MethodPost, "request method")
	flag.StringVar(&cfg.path, "path", "", "endpoint path, with its query string")
	flag.StringVar(&cfg.body, "body", "", "request body (JSON)")
	flag.StringVar(&cfg.bodyFile, "body-file", "", "file holding the request body")
	flag.StringVar(&cfg.contentType, "content-type", "application/json", "Content-Type of the request body")
	flag.StringVar(&cfg.out, "out", "", "fixture path prefix, for example openai/testdata/chat_completions/basic")
	flag.IntVar(&cfg.wantStatus, "want-status", http.StatusOK, "expected response status")
	flag.DurationVar(&cfg.timeout, "timeout", 5*time.Minute, "overall timeout")
	flag.Parse()
	if err := record(&cfg); err != nil {
		log.Fatal(err)
	}
}

// record runs the recording under an interrupt handler and the overall timeout.
func record(cfg *config) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, cfg.timeout)
	defer cancel()
	return run(ctx, cfg)
}

func run(ctx context.Context, cfg *config) error {
	if cfg.path == "" || cfg.out == "" {
		return errors.New("-path and -out are required")
	}
	if (cfg.server == "") == (cfg.model == "") {
		return errors.New("give exactly one of -server and -model")
	}
	body, err := requestBody(cfg)
	if err != nil {
		return err
	}

	base, socket := cfg.server, cfg.socket
	if cfg.model != "" {
		modelPath, err := harness.ModelPath(ctx, cfg.model)
		if err != nil {
			return err
		}
		srv, err := harness.Start(ctx, harness.Options{Model: modelPath, Args: strings.Fields(cfg.serverArgs)})
		if err != nil {
			return err
		}
		defer func() {
			if err := srv.Stop(); err != nil {
				log.Print(err)
			}
		}()
		base, socket = srv.URL, srv.Socket
	}

	opts := []llama.ClientOption{llama.WithRetry(llama.RetryPolicy{MaxAttempts: 1})}
	if socket != "" {
		opts = append(opts, llama.WithUnixSocket(socket))
	}
	if cfg.apiKeyEnv != "" {
		key := os.Getenv(cfg.apiKeyEnv)
		if key == "" {
			return fmt.Errorf("environment variable %s is empty", cfg.apiKeyEnv)
		}
		opts = append(opts, llama.WithAPIKey(key))
	}
	c, err := llama.New(base, opts...)
	if err != nil {
		return err
	}
	defer c.HTTPClient().CloseIdleConnections()

	var reqBody any
	if body != nil {
		reqBody = &llama.Body{ContentType: cfg.contentType, Data: body}
	}
	req, err := c.NewRequest(ctx, cfg.method, cfg.path, reqBody)
	if err != nil {
		return err
	}
	resp, err := c.HTTPClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading the response: %w", err)
	}
	if resp.StatusCode != cfg.wantStatus {
		return fmt.Errorf("%s %s: status %d, want %d; nothing written\n%s", cfg.method, cfg.path, resp.StatusCode, cfg.wantStatus, snippet(respBody))
	}

	if err := os.MkdirAll(filepath.Dir(cfg.out), 0o755); err != nil {
		return err
	}
	var written []string
	if body != nil {
		name := cfg.out + ".req.json"
		if !strings.HasPrefix(cfg.contentType, "application/json") {
			name = cfg.out + ".req.bin"
		}
		if err := os.WriteFile(name, body, 0o644); err != nil {
			return err
		}
		written = append(written, name)
	}
	name := cfg.out + ".resp" + extension(resp.Header.Get("Content-Type"))
	if err := os.WriteFile(name, respBody, 0o644); err != nil {
		return err
	}
	written = append(written, name)
	fmt.Fprintf(os.Stderr, "%s %s: %d %s, wrote %s\n", cfg.method, cfg.path, resp.StatusCode, resp.Header.Get("Content-Type"), strings.Join(written, ", "))
	return nil
}

// requestBody returns the body given by -body or -body-file, or nil. A JSON body must be valid JSON.
func requestBody(cfg *config) ([]byte, error) {
	var body []byte
	switch {
	case cfg.body != "" && cfg.bodyFile != "":
		return nil, errors.New("give at most one of -body and -body-file")
	case cfg.body != "":
		body = []byte(cfg.body)
	case cfg.bodyFile != "":
		var err error
		if body, err = os.ReadFile(cfg.bodyFile); err != nil {
			return nil, err
		}
	default:
		return nil, nil
	}
	if strings.HasPrefix(cfg.contentType, "application/json") {
		if !jsontext.Value(body).IsValid() {
			return nil, errors.New("request body is not valid JSON")
		}
	}
	return body, nil
}

// extension returns the fixture extension for a response Content-Type.
func extension(contentType string) string {
	mt, _, _ := mime.ParseMediaType(contentType)
	switch mt {
	case "application/json":
		return ".json"
	case "text/event-stream":
		return ".sse"
	case "text/plain":
		return ".txt"
	default:
		return ".bin"
	}
}

func snippet(b []byte) string {
	b = bytes.TrimSpace(b)
	if len(b) > 512 {
		b = b[:512]
	}
	return string(b)
}
