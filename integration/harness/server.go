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

// Package harness runs a real llama-server for integration tests and observes the HTTP traffic of a client
// without changing it.
//
// Start launches the binary named by LLAMA_SERVER_BIN (default: llama-server on PATH) on a free local port or on
// a UNIX socket, waits until GET /health answers 200 and returns a Server; Stop ends the process. The server runs
// under a supervisor that stops it when the starting process exits for any reason (see supervise.go), so a test
// that panics or is killed leaves no server behind. Server.Client
// returns a llama.Client whose transport is an Observer, a RoundTripper that records each exchange (method, path,
// query, headers and whether the response body was closed) and forwards it to a real http.Transport.
//
// Model returns the path of a model file listed in integration/models.yaml, downloading it from Hugging Face at
// the pinned revision and verifying its sha256 on first use.
package harness

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	llama "github.com/zchee/go-llama.cpp"
)

// EnvServerBin names the environment variable that selects the llama-server binary.
const EnvServerBin = "LLAMA_SERVER_BIN"

// DefaultStartTimeout bounds the wait for a started server to become ready.
const DefaultStartTimeout = 3 * time.Minute

// stopTimeout bounds the wait for the process to exit after an interrupt before it is killed.
const stopTimeout = 15 * time.Second

// Options describes a llama-server process.
type Options struct {
	// Model is the model file passed with --model. Leave it empty for router mode.
	Model string
	// ModelsDir is the model directory of router mode, passed with --models-dir. Like Model, it is reached
	// through a relative symlink so that responses carry no local path.
	ModelsDir string
	// Online lets the server reach the network, for example to download a model through POST /models. By
	// default the server runs with --offline.
	Online bool
	// Args are further command-line flags, for example "--metrics" or "--ctx-size", "512".
	Args []string
	// APIKey starts the server with --api-key.
	APIKey string
	// UnixSocket makes the server listen on a UNIX socket instead of a TCP port.
	UnixSocket bool
	// StartTimeout bounds the wait for readiness; zero selects DefaultStartTimeout.
	StartTimeout time.Duration
}

// Server is a running llama-server process.
type Server struct {
	// URL is the base URL of the server. With a UNIX socket it is "http://localhost" and Socket is set.
	URL string
	// Socket is the UNIX socket path, empty for a TCP server.
	Socket string
	// APIKey is the key the server was started with, empty when it needs none.
	APIKey string

	dir       string
	logPath   string
	cmd       *exec.Cmd
	ownerPipe *os.File
	pid       int
	done      chan struct{}
	waitErr   error
	stopOnce  sync.Once
	stopErr   error
}

// Bin returns the llama-server binary to run.
func Bin() string {
	if bin := os.Getenv(EnvServerBin); bin != "" {
		return bin
	}
	return "llama-server"
}

// Start launches a server and waits until it is ready. The caller must call Stop.
func Start(ctx context.Context, opts Options) (*Server, error) {
	bin, err := exec.LookPath(Bin())
	if err != nil {
		return nil, fmt.Errorf("harness: llama-server binary: %w (set %s)", err, EnvServerBin)
	}
	// os.MkdirTemp keeps the socket path short: UNIX socket paths are limited to about 100 bytes.
	temp, err := filepath.Abs(os.TempDir())
	if err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp(temp, "llama-")
	if err != nil {
		return nil, err
	}
	s := &Server{dir: dir, logPath: filepath.Join(dir, "server.log"), APIKey: opts.APIKey, done: make(chan struct{})}

	if err := os.Mkdir(s.CacheDir(), 0o700); err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}

	args := []string{"--no-webui", "--log-colors", "off"}
	if !opts.Online {
		args = append(args, "--offline")
	}
	// The server reports the model paths it was given (for example in the "model" member of completions and in
	// /props), and recorded fixtures must not carry a local path. Relative symlinks inside the working directory
	// of the process keep the reported paths free of user directories.
	if opts.Model != "" {
		if err := os.Mkdir(filepath.Join(dir, "models"), 0o755); err != nil {
			_ = os.RemoveAll(dir)
			return nil, err
		}
		rel, err := linkInto(dir, opts.Model, filepath.Join("models", filepath.Base(opts.Model)))
		if err != nil {
			_ = os.RemoveAll(dir)
			return nil, err
		}
		args = append(args, "--model", rel)
	}
	if opts.ModelsDir != "" {
		rel, err := linkInto(dir, opts.ModelsDir, "models-dir")
		if err != nil {
			_ = os.RemoveAll(dir)
			return nil, err
		}
		args = append(args, "--models-dir", rel)
	}
	if opts.UnixSocket {
		s.Socket = filepath.Join(dir, "server.sock")
		s.URL = "http://localhost"
		args = append(args, "--host", s.Socket)
	} else {
		port, err := freePort()
		if err != nil {
			_ = os.RemoveAll(dir)
			return nil, err
		}
		s.URL = "http://127.0.0.1:" + strconv.Itoa(port)
		args = append(args, "--host", "127.0.0.1", "--port", strconv.Itoa(port))
	}
	if opts.APIKey != "" {
		args = append(args, "--api-key", opts.APIKey)
	}
	args = append(args, opts.Args...)

	self, err := os.Executable()
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, fmt.Errorf("harness: locating the supervisor binary: %w", err)
	}
	logFile, err := os.Create(s.logPath)
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	// The read end becomes the supervisor's standard input; this process keeps the only write end (os.Pipe sets
	// close-on-exec, so no other child inherits it). See init in supervise.go.
	ownerRead, ownerWrite, err := os.Pipe()
	if err != nil {
		_ = logFile.Close()
		_ = os.RemoveAll(dir)
		return nil, err
	}
	// The processes write their output straight into the file, so no goroutine copies pipes.
	s.cmd = exec.Command(self, append([]string{bin}, args...)...)
	s.cmd.Env = append(os.Environ(), envSupervise+"=1", "LLAMA_CACHE="+s.CacheDir())
	s.cmd.Stdin = ownerRead
	s.cmd.Stdout = logFile
	s.cmd.Stderr = logFile
	s.cmd.Dir = dir
	s.cmd.SysProcAttr = sysProcAttr()
	err = s.cmd.Start()
	_ = logFile.Close()
	_ = ownerRead.Close()
	if err != nil {
		_ = ownerWrite.Close()
		_ = os.RemoveAll(dir)
		return nil, fmt.Errorf("harness: starting the supervisor of %s: %w", bin, err)
	}
	s.ownerPipe = ownerWrite
	go func() {
		s.waitErr = s.cmd.Wait()
		close(s.done)
	}()

	timeout := opts.StartTimeout
	if timeout <= 0 {
		timeout = DefaultStartTimeout
	}
	if err := s.waitReady(ctx, timeout); err != nil {
		// Stop removes the log, so its tail is read first.
		tail := s.LogTail(4096)
		stopErr := s.Stop()
		return nil, errors.Join(fmt.Errorf("harness: server %s did not become ready: %w\n--- server log tail ---\n%s", s.URL, err, tail), stopErr)
	}
	data, err := os.ReadFile(filepath.Join(dir, pidFile))
	if err == nil {
		s.pid, err = strconv.Atoi(string(data))
	}
	if err != nil {
		tail := s.LogTail(4096)
		return nil, errors.Join(fmt.Errorf("harness: reading the llama-server PID: %w\n--- server log tail ---\n%s", err, tail), s.Stop())
	}
	return s, nil
}

// StartT starts a server for the test and stops it when the test and its subtests end.
func StartT(t testing.TB, opts Options) *Server {
	t.Helper()
	s, err := Start(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Stop(); err != nil {
			t.Errorf("stopping llama-server: %v", err)
		}
	})
	return s
}

// waitReady polls /health until it answers 200, the process exits, or the timeout passes.
func (s *Server) waitReady(ctx context.Context, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	exited := make(chan struct{})
	go func() {
		select {
		case <-s.done:
			cancel()
		case <-ctx.Done():
		}
		close(exited)
	}()
	defer func() { cancel(); <-exited }()

	tr := s.transport()
	defer tr.CloseIdleConnections()
	c, err := llama.New(s.URL, llama.WithHTTPClient(&http.Client{Transport: tr}))
	if err != nil {
		return err
	}
	err = c.WaitReady(ctx)
	select {
	case <-s.done:
		return fmt.Errorf("process exited: %w", s.waitErr)
	default:
	}
	return err
}

// transport returns a fresh transport that reaches the server, over its UNIX socket when it has one.
func (s *Server) transport() *http.Transport {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.Proxy = nil
	if s.Socket != "" {
		var d net.Dialer
		tr.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return d.DialContext(ctx, "unix", s.Socket)
		}
	}
	return tr
}

// Client returns a client of the server whose requests pass through a new Observer. The API key is not set; pass
// llama.WithAPIKey(s.APIKey) when the test needs it. Idle connections are closed when the test ends.
func (s *Server) Client(t testing.TB, opts ...llama.ClientOption) (*llama.Client, *Observer) {
	t.Helper()
	tr := s.transport()
	obs := NewObserver(tr)
	t.Cleanup(tr.CloseIdleConnections)
	opts = append([]llama.ClientOption{llama.WithHTTPClient(&http.Client{Transport: obs})}, opts...)
	c, err := llama.New(s.URL, opts...)
	if err != nil {
		t.Fatalf("llama.New(%s): %v", s.URL, err)
	}
	return c, obs
}

// CacheDir returns the private llama-server cache directory removed by Stop.
// It is separate from the harness cache for pinned model assets.
func (s *Server) CacheDir() string { return filepath.Join(s.dir, "cache") }

// PID returns the process ID of llama-server.
func (s *Server) PID() int {
	return s.pid
}

// Stop ends the server through its supervisor: closing the owner pipe makes the supervisor interrupt llama-server
// and kill it if it does not exit in time. Stop then waits for the supervisor and removes the server's files. It
// is safe to call more than once.
func (s *Server) Stop() error {
	s.stopOnce.Do(func() {
		_ = s.ownerPipe.Close()
		// The supervisor needs at most stopTimeout; the margin covers its own exit.
		timer := time.NewTimer(stopTimeout + 10*time.Second)
		select {
		case <-s.done:
			timer.Stop()
		case <-timer.C:
			// Start can fail before it records s.pid. The server group is separate from the supervisor's.
			if data, err := os.ReadFile(filepath.Join(s.dir, pidFile)); err == nil {
				if pid, err := strconv.Atoi(string(data)); err == nil && pid > 1 {
					killGroup(pid)
				}
			}
			killGroup(s.cmd.Process.Pid)
			<-s.done
			s.stopErr = errors.New("harness: the llama-server supervisor did not exit and its process group was killed")
		}
		if ee, ok := errors.AsType[*exec.ExitError](s.waitErr); ok && ee.ExitCode() == exitKilled && s.stopErr == nil {
			s.stopErr = fmt.Errorf("harness: llama-server did not exit within %v of an interrupt and was killed", stopTimeout)
		}
		if err := os.RemoveAll(s.dir); err != nil && s.stopErr == nil {
			s.stopErr = err
		}
	})
	return s.stopErr
}

// Exited reports whether the process has exited.
func (s *Server) Exited() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

// LogTail returns the last n bytes of the server output.
func (s *Server) LogTail(n int) string {
	data, err := os.ReadFile(s.logPath)
	if err != nil {
		return fmt.Sprintf("(log unavailable: %v)", err)
	}
	if len(data) > n {
		data = data[len(data)-n:]
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			data = data[i+1:]
		}
	}
	return string(data)
}

// linkInto creates a symlink to target at the relative path rel under dir and returns rel.
func linkInto(dir, target, rel string) (string, error) {
	abs, err := filepath.Abs(target)
	if err != nil {
		return "", err
	}
	if err := os.Symlink(abs, filepath.Join(dir, rel)); err != nil {
		return "", err
	}
	return rel, nil
}

// freePort returns a TCP port on 127.0.0.1 that was free a moment ago.
func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}
