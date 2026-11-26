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

package integration

import (
	"bufio"
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	llama "github.com/zchee/go-llama.cpp"
	"github.com/zchee/go-llama.cpp/integration/harness"
)

// endpointsFile lists the endpoints the server README documents, one "METHOD PATH" key per line.
const endpointsFile = "../testdata/endpoints.v1.1.0.txt"

// contractCase proves that one documented endpoint is reached through its typed method.
//
// The key of a case is the line of endpointsFile it covers: the method ("WebSocket" for a WebSocket upgrade), a
// space, and the documented path, in which "{name}" stands for one path segment and a query string names the
// parameters that tell routes with the same path apart (for example "POST /slots/{id_slot}?action=save").
type contractCase struct {
	// server is the shared server the case runs against.
	server *serverPreset
	// clientOptions are added to the observed client, for example llama.WithAPIKey.
	clientOptions []llama.ClientOption
	// call invokes the typed method of the endpoint and returns its error. It must not use llama.Call,
	// llama.Stream or llama.CallRaw; the contract test rejects files that do.
	call func(t *testing.T, c *llama.Client) error
	// query lists query parameters, besides those of the key, that the request must carry, for example the
	// "model" parameter of a GET routed by a router.
	query url.Values
}

// contractTables returns the contract tables of the endpoint groups, keyed by the file that holds them.
func contractTables() map[string]map[string]contractCase {
	return map[string]map[string]contractCase{
		"contract_native_inference_test.go": nativeInferenceContracts,
		"contract_native_admin_test.go":     nativeAdminContracts,
		"contract_audio_decision_test.go":   audioDecisionContracts,
		"contract_openai_test.go":           openaiContracts,
		"contract_anthropic_test.go":        anthropicContracts,
		"contract_realtime_test.go":         realtimeContracts,
	}
}

// allContracts merges the contract tables and fails the test when two tables cover the same endpoint.
func allContracts(t *testing.T) map[string]contractCase {
	t.Helper()
	all := map[string]contractCase{}
	owner := map[string]string{}
	tables := contractTables()
	for _, file := range slices.Sorted(maps.Keys(tables)) {
		for key, c := range tables[file] {
			if prev, ok := owner[key]; ok {
				t.Errorf("endpoint %q has contract cases in both %s and %s", key, prev, file)
				continue
			}
			owner[key] = file
			all[key] = c
		}
	}
	return all
}

// documentedEndpoints reads the endpoint keys of endpointsFile.
func documentedEndpoints(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile(endpointsFile)
	if err != nil {
		t.Fatalf("reading the endpoint list: %v", err)
	}
	var keys []string
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		keys = append(keys, line)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(keys) != 38 {
		t.Fatalf("%s lists %d endpoints, want 38; regenerate it with go generate", endpointsFile, len(keys))
	}
	return keys
}

// TestContractCoverage checks that the contract tables cover exactly the documented endpoints.
func TestContractCoverage(t *testing.T) {
	documented := documentedEndpoints(t)
	contracts := allContracts(t)

	var missing []string
	for _, key := range documented {
		if _, ok := contracts[key]; !ok {
			missing = append(missing, key)
		}
	}
	var unknown []string
	for _, key := range slices.Sorted(maps.Keys(contracts)) {
		if !slices.Contains(documented, key) {
			unknown = append(unknown, key)
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d of %d documented endpoints have no contract case:\n  %s", len(missing), len(documented), strings.Join(missing, "\n  "))
	}
	if len(unknown) > 0 {
		t.Errorf("%d contract cases name no documented endpoint (keys must match %s exactly):\n  %s", len(unknown), endpointsFile, strings.Join(unknown, "\n  "))
	}
}

// TestContracts runs every contract case against the real server and checks the route the request took.
func TestContracts(t *testing.T) {
	contracts := allContracts(t)
	for _, key := range slices.Sorted(maps.Keys(contracts)) {
		c := contracts[key]
		t.Run(key, func(t *testing.T) {
			route, err := parseEndpointKey(key)
			if err != nil {
				t.Fatal(err)
			}
			if c.server == nil || c.call == nil {
				t.Fatalf("contract case %q needs a server and a call", key)
			}
			srv := c.server.server(t)
			client, obs := srv.Client(t, c.clientOptions...)
			if err := c.call(t, client); err != nil {
				t.Errorf("typed method for %s failed against the real server: %v", key, err)
			}
			exchanges := obs.Exchanges()
			if len(exchanges) == 0 {
				t.Fatalf("the typed method for %s sent no request", key)
			}
			for _, ex := range exchanges {
				if route.matches(ex, c.query) {
					return
				}
			}
			t.Errorf("no request matched %s (query %v); observed:\n  %s", key, c.query, describe(exchanges))
		})
	}
}

// route is a parsed endpoint key.
type route struct {
	method    string
	websocket bool
	path      *regexp.Regexp
	query     url.Values
}

var placeholder = regexp.MustCompile(`\{[^/{}]+\}`)

// parseEndpointKey parses a key such as "POST /slots/{id_slot}?action=save" or "WebSocket /v1/realtime".
func parseEndpointKey(key string) (*route, error) {
	method, target, ok := strings.Cut(key, " ")
	if !ok {
		return nil, fmt.Errorf("endpoint key %q has no path", key)
	}
	p, rawQuery, _ := strings.Cut(target, "?")
	query, err := url.ParseQuery(rawQuery)
	if err != nil {
		return nil, fmt.Errorf("endpoint key %q: %w", key, err)
	}
	var pattern strings.Builder
	pattern.WriteString("^")
	last := 0
	for _, loc := range placeholder.FindAllStringIndex(p, -1) {
		pattern.WriteString(regexp.QuoteMeta(p[last:loc[0]]))
		pattern.WriteString("[^/]+")
		last = loc[1]
	}
	pattern.WriteString(regexp.QuoteMeta(p[last:]))
	pattern.WriteString("$")
	r := &route{method: method, path: regexp.MustCompile(pattern.String()), query: query}
	if method == "WebSocket" {
		r.method = "GET"
		r.websocket = true
	}
	return r, nil
}

// matches reports whether an observed exchange took the route and carried the required query parameters.
func (r *route) matches(ex *harness.Exchange, extra url.Values) bool {
	if ex.Method != r.method || !r.path.MatchString(ex.Path) || ex.Upgrade() != r.websocket {
		return false
	}
	for _, q := range []url.Values{r.query, extra} {
		for k, want := range q {
			for _, v := range want {
				if !slices.Contains(ex.Query[k], v) {
					return false
				}
			}
		}
	}
	return true
}

func describe(exchanges []*harness.Exchange) string {
	lines := make([]string, len(exchanges))
	for i, ex := range exchanges {
		lines[i] = fmt.Sprintf("%s %s?%s upgrade=%t status=%d err=%v", ex.Method, ex.Path, ex.Query.Encode(), ex.Upgrade(), ex.StatusCode, ex.Err)
	}
	return strings.Join(lines, "\n  ")
}

func TestParseEndpointKey(t *testing.T) {
	tests := map[string]struct {
		key       string
		method    string
		websocket bool
		match     []string
		noMatch   []string
		query     url.Values
	}{
		"success: plain": {
			key: "GET /health", method: "GET",
			match: []string{"/health"}, noMatch: []string{"/v1/health", "/health/x"},
		},
		"success: placeholder and discriminator": {
			key: "POST /slots/{id_slot}?action=save", method: "POST",
			match: []string{"/slots/0", "/slots/12"}, noMatch: []string{"/slots", "/slots/0/x"},
			query: url.Values{"action": {"save"}},
		},
		"success: WebSocket": {
			key: "WebSocket /v1/realtime", method: "GET", websocket: true,
			match: []string{"/v1/realtime"},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			r, err := parseEndpointKey(tt.key)
			if err != nil {
				t.Fatal(err)
			}
			if r.method != tt.method || r.websocket != tt.websocket {
				t.Errorf("method = %s websocket = %t, want %s %t", r.method, r.websocket, tt.method, tt.websocket)
			}
			for _, p := range tt.match {
				if !r.path.MatchString(p) {
					t.Errorf("%s does not match %s", tt.key, p)
				}
			}
			for _, p := range tt.noMatch {
				if r.path.MatchString(p) {
					t.Errorf("%s matches %s", tt.key, p)
				}
			}
			if len(tt.query) > 0 && r.query.Encode() != tt.query.Encode() {
				t.Errorf("query = %v, want %v", r.query, tt.query)
			}
		})
	}
}

// lowLevelBridge names the functions of the root package that reach any route without a typed method.
var lowLevelBridge = []string{"Call", "Stream", "CallRaw"}

// TestContractFilesUseTypedMethods parses the contract files and fails for every reference to the low-level
// bridge, which would satisfy a contract without a typed method.
func TestContractFilesUseTypedMethods(t *testing.T) {
	files, err := filepath.Glob("contract*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	for file := range contractTables() {
		if !slices.Contains(files, file) {
			t.Errorf("contract table file %s does not exist", file)
		}
	}
	fset := token.NewFileSet()
	for _, file := range files {
		f, err := parser.ParseFile(fset, file, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parsing %s: %v", file, err)
		}
		for _, v := range bridgeReferences(fset, f) {
			t.Errorf("%s: contract files must call typed methods, not the low-level llama.%s", v.pos, v.name)
		}
	}
}

type bridgeReference struct {
	pos  token.Position
	name string
}

// bridgeReferences returns the references to the low-level bridge functions of the root package in f, under
// whatever name the file imports the package with. A dot import is reported as well.
func bridgeReferences(fset *token.FileSet, f *ast.File) []bridgeReference {
	const rootPath = "github.com/zchee/go-llama.cpp"
	var names []string
	var refs []bridgeReference
	for _, imp := range f.Imports {
		p, err := strconv.Unquote(imp.Path.Value)
		if err != nil || p != rootPath {
			continue
		}
		switch {
		case imp.Name == nil:
			names = append(names, "llama")
		case imp.Name.Name == ".":
			refs = append(refs, bridgeReference{pos: fset.Position(imp.Pos()), name: "* (dot import)"})
		case imp.Name.Name != "_":
			names = append(names, imp.Name.Name)
		}
	}
	ast.Inspect(f, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if id, ok := sel.X.(*ast.Ident); ok && slices.Contains(names, id.Name) && slices.Contains(lowLevelBridge, sel.Sel.Name) {
			refs = append(refs, bridgeReference{pos: fset.Position(sel.Pos()), name: sel.Sel.Name})
		}
		return true
	})
	return refs
}

func TestBridgeReferences(t *testing.T) {
	tests := map[string]struct {
		src  string
		want []string
	}{
		"success: typed method only": {
			src:  "package p\nimport \"github.com/zchee/go-llama.cpp\"\nfunc f(c *llama.Client) { c.Health(nil) }\n",
			want: nil,
		},
		"error: generic call and stream": {
			src:  "package p\nimport \"github.com/zchee/go-llama.cpp\"\nvar _ = llama.Call[int]\nvar _ = llama.Stream[int]\nvar _ = llama.CallRaw\n",
			want: []string{"Call", "Stream", "CallRaw"},
		},
		"error: renamed import": {
			src:  "package p\nimport l \"github.com/zchee/go-llama.cpp\"\nvar _ = l.CallRaw\n",
			want: []string{"CallRaw"},
		},
		"error: dot import": {
			src:  "package p\nimport . \"github.com/zchee/go-llama.cpp\"\n",
			want: []string{"* (dot import)"},
		},
		"success: same names on another package": {
			src:  "package p\nimport llama \"example.com/other\"\nvar _ = llama.Call\n",
			want: nil,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "x.go", tt.src, parser.SkipObjectResolution)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, r := range bridgeReferences(fset, f) {
				got = append(got, r.name)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("bridgeReferences = %v, want %v", got, tt.want)
			}
		})
	}
}
