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
	"go/build"
	"go/constant"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	gocmp "github.com/google/go-cmp/cmp"

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

// contractRegistration is one call of registerContracts.
type contractRegistration struct {
	// file is the base name of the file that registered the cases.
	file  string
	cases map[string]contractCase
}

// contractRegistrations holds every registration, in package initialisation order.
var contractRegistrations []contractRegistration

// registerContracts records contract cases for the endpoints of one contract_<name>_test.go file. It is meant as a
// package-level variable initialiser:
//
//	var _ = registerContracts(map[string]contractCase{
//		"GET /health": {server: presetStories260K, call: func(t *testing.T, c *llama.Client) error { ... }},
//	})
//
// The file is taken from the caller, so a registration cannot claim another file; TestContractFilesUseTypedMethods
// checks that it is one of the contract files it parses.
func registerContracts(cases map[string]contractCase) struct{} {
	file := "unknown"
	if _, path, _, ok := runtime.Caller(1); ok {
		file = filepath.Base(path)
	}
	contractRegistrations = append(contractRegistrations, contractRegistration{file: file, cases: cases})
	return struct{}{}
}

// mergeContracts merges registrations into one table. problems names every endpoint that two registrations
// cover, with both files.
func mergeContracts(regs []contractRegistration) (all map[string]contractCase, problems []string) {
	all = map[string]contractCase{}
	owner := map[string]string{}
	for _, reg := range regs {
		for _, key := range slices.Sorted(maps.Keys(reg.cases)) {
			if prev, ok := owner[key]; ok {
				problems = append(problems, fmt.Sprintf("endpoint %q has contract cases in both %s and %s", key, prev, reg.file))
				continue
			}
			owner[key] = reg.file
			all[key] = reg.cases[key]
		}
	}
	return all, problems
}

// allContracts merges the registered contract cases and fails the test when two registrations cover the same
// endpoint.
func allContracts(t *testing.T) map[string]contractCase {
	t.Helper()
	all, problems := mergeContracts(contractRegistrations)
	for _, p := range problems {
		t.Error(p)
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

// matches reports whether an observed exchange took the route and carried the required query parameters, each
// with exactly the required values.
func (r *route) matches(ex harness.Exchange, extra url.Values) bool {
	if ex.Method != r.method || !r.path.MatchString(ex.Path) || ex.Upgrade() != r.websocket {
		return false
	}
	// The server keeps only one value of a repeated query parameter (the last, in get_params of server-http.cpp),
	// so a parameter the key names must be sent exactly as named, not among other values.
	for _, q := range []url.Values{r.query, extra} {
		for k, want := range q {
			if !slices.Equal(ex.Query[k], want) {
				return false
			}
		}
	}
	return true
}

func describe(exchanges []harness.Exchange) string {
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

// TestContractFilesUseTypedMethods rejects low-level bridge references and unresolved calls reachable from a contract.
func TestContractFilesUseTypedMethods(t *testing.T) {
	files, err := contractFiles(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, problem := range unlistedRegistrations(contractRegistrations, files) {
		t.Error(problem)
	}
	pkg, err := build.Default.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var syntax []*ast.File
	for _, path := range append(pkg.GoFiles, pkg.TestGoFiles...) {
		f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		syntax = append(syntax, f)
	}
	for _, problem := range contractBridgeChains(t, fset, syntax, files, integrationImporter(t, fset)) {
		t.Error(problem)
	}
}

func integrationImporter(t *testing.T, fset *token.FileSet) types.Importer {
	t.Helper()
	// go/importer alone cannot find modules. Resolve their export data with the same workspace as go test.
	out, err := exec.CommandContext(t.Context(), "go", "list", "-deps", "-test", "-export", "-f", `{{if .Export}}{{.ImportPath}}{{"\t"}}{{.Export}}{{end}}`, ".").Output()
	if err != nil {
		t.Fatal(err)
	}
	paths := map[string]string{}
	for line := range strings.Lines(string(out)) {
		name, path, _ := strings.Cut(strings.TrimSpace(line), "\t")
		paths[name] = path
	}
	return importer.ForCompiler(fset, "gc", func(path string) (io.ReadCloser, error) { return os.Open(paths[path]) })
}

// contractCallee returns the declared object of a direct call, including generic instantiations.
func contractCallee(info *types.Info, expr ast.Expr) types.Object {
	switch e := ast.Unparen(expr).(type) {
	case *ast.Ident:
		return info.Uses[e]
	case *ast.SelectorExpr:
		return info.Uses[e.Sel]
	case *ast.IndexExpr:
		if info.Types[e.Index].IsType() {
			return contractCallee(info, e.X)
		}
	case *ast.IndexListExpr:
		return contractCallee(info, e.X)
	}
	return nil
}

// localContractFunctions identifies local function literals and standard context cancellation functions whose
// bindings are only ever called, never reassigned or taken by address.
func localContractFunctions(info *types.Info, pkg *types.Package, syntax []*ast.File) map[types.Object]bool {
	known := map[types.Object]bool{}
	called := map[*ast.Ident]bool{}
	bind := func(id *ast.Ident, rhs ast.Expr) {
		obj := info.Defs[id]
		if _, ok := ast.Unparen(rhs).(*ast.FuncLit); ok && obj != nil && obj.Parent() != pkg.Scope() {
			known[obj] = true
		}
	}
	for _, file := range syntax {
		ast.Inspect(file, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.CallExpr:
				if id, ok := ast.Unparen(n.Fun).(*ast.Ident); ok {
					called[id] = true
				}
			case *ast.AssignStmt:
				if n.Tok != token.DEFINE {
					break
				}
				if len(n.Lhs) == len(n.Rhs) {
					for i, lhs := range n.Lhs {
						bind(lhs.(*ast.Ident), n.Rhs[i])
					}
				} else if call, ok := n.Rhs[0].(*ast.CallExpr); ok && len(n.Lhs) == 2 {
					if fn, ok := contractCallee(info, call.Fun).(*types.Func); ok && fn.Pkg() != nil && fn.Pkg().Path() == "context" && fn.Name() == "WithTimeout" {
						if obj := info.Defs[n.Lhs[1].(*ast.Ident)]; obj != nil {
							known[obj] = true
						}
					}
				}
			case *ast.ValueSpec:
				if len(n.Names) == len(n.Values) {
					for i, id := range n.Names {
						bind(id, n.Values[i])
					}
				}
			}
			return true
		})
	}
	for id, obj := range info.Uses {
		if !called[id] {
			delete(known, obj)
		}
	}
	return known
}

func contractBridgeChains(t *testing.T, fset *token.FileSet, syntax []*ast.File, roots []string, imp types.Importer) []string {
	t.Helper()
	info := &types.Info{Defs: map[*ast.Ident]types.Object{}, Uses: map[*ast.Ident]types.Object{}, Types: map[ast.Expr]types.TypeAndValue{}}
	cfg := types.Config{Importer: imp}
	pkg, err := cfg.Check("github.com/zchee/go-llama.cpp/integration", fset, syntax, info)
	if err != nil {
		t.Fatal(err)
	}
	known := localContractFunctions(info, pkg, syntax)
	caseType := pkg.Scope().Lookup("contractCase").Type()
	fields := caseType.Underlying().(*types.Struct)
	register := pkg.Scope().Lookup("registerContracts")
	bodies := map[types.Object]ast.Node{}
	var problems []string
	for _, file := range syntax {
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Body != nil {
					bodies[info.Defs[d.Name]] = d.Body
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					if v, ok := spec.(*ast.ValueSpec); ok && len(v.Names) == len(v.Values) {
						for i, id := range v.Names {
							bodies[info.Defs[id]] = v.Values[i]
						}
					}
				}
			}
		}
	}
	resolved := func(fun ast.Expr) bool {
		switch obj := contractCallee(info, fun).(type) {
		case *types.Func:
			return obj.Pkg() != pkg || bodies[obj.Origin()] != nil
		case *types.Builtin:
			return true
		case *types.Var:
			return known[obj]
		default:
			_, literal := ast.Unparen(fun).(*ast.FuncLit)
			return literal
		}
	}
	for _, file := range syntax {
		name := fset.Position(file.Pos()).Filename
		if !slices.Contains(roots, name) {
			continue
		}
		for _, ref := range bridgeReferences(fset, file) {
			if ref.name == "* (dot import)" {
				problems = append(problems, ref.pos.String()+": forbidden dot import")
			}
		}
		seen := map[types.Object]bool{}
		var walk func(ast.Node, string)
		walk = func(body ast.Node, chain string) {
			ast.Inspect(body, func(n ast.Node) bool {
				if kv, ok := n.(*ast.KeyValueExpr); ok && types.Identical(info.TypeOf(kv.Value), caseType) {
					key := types.ExprString(kv.Key)
					if v := info.Types[kv.Key].Value; v != nil && v.Kind() == constant.String {
						key = constant.StringVal(v)
					}
					walk(kv.Value, chain+": "+key)
					return false
				}
				if call, ok := n.(*ast.CallExpr); ok && !info.Types[call.Fun].IsType() && !resolved(call.Fun) {
					problems = append(problems, chain+" -> unresolved call "+types.ExprString(call))
				}
				if lit, ok := n.(*ast.CompositeLit); ok && types.Identical(info.TypeOf(lit), caseType) {
					// Server presets and options are not callbacks; exclude their initializers.
					for i, elem := range lit.Elts {
						field := ""
						if kv, ok := elem.(*ast.KeyValueExpr); ok {
							field, elem = kv.Key.(*ast.Ident).Name, kv.Value
						} else {
							field = fields.Field(i).Name()
						}
						if field == "call" {
							if !resolved(elem) {
								problems = append(problems, chain+" -> unresolved callback "+types.ExprString(elem))
							}
							walk(elem, chain)
						}
					}
					return false
				}
				id, ok := n.(*ast.Ident)
				if !ok {
					return true
				}
				obj := info.Uses[id]
				if fn, ok := obj.(*types.Func); ok {
					if fn.Pkg() != nil && fn.Pkg().Path() == "github.com/zchee/go-llama.cpp" && fn.Signature().Recv() == nil && slices.Contains(lowLevelBridge, fn.Name()) {
						problems = append(problems, chain+" -> llama."+fn.Name())
						return true
					}
					obj = fn.Origin()
				}
				if child := bodies[obj]; child != nil && !seen[obj] {
					seen[obj] = true
					label := obj.Name()
					if fn, ok := obj.(*types.Func); ok && fn.Signature().Recv() != nil {
						label = "(" + types.TypeString(fn.Signature().Recv().Type(), func(*types.Package) string { return "" }) + ")." + label
					}
					walk(child, chain+" -> "+label)
					delete(seen, obj)
				}
				return true
			})
		}
		ast.Inspect(file, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if id, ok := call.Fun.(*ast.Ident); ok && info.Uses[id] == register {
				for _, arg := range call.Args {
					walk(arg, name)
				}
			}
			return true
		})
	}
	slices.Sort(problems)
	return slices.Compact(problems)
}

func TestContractBridgeChains(t *testing.T) {
	const root = "contract_sample_test.go"
	const helpers = `package integration
import l "github.com/zchee/go-llama.cpp"
type runner struct{}
type clean struct{}
func (runner) invoke() { _, _ = l.CallRaw(nil, nil, "GET", "/health", nil) }
func (clean) invoke() {}
func indirect() { runner{}.invoke() }
func unused() { _ = l.Stream[int] }
type invokes interface { invoke() }
var dynamic invokes = runner{}
var later func()
var holder struct{ run func() }
func init() {
	later = func() { _, _ = l.CallRaw(nil, nil, "GET", "/health", nil) }
	holder.run = later
}
`
	fset := token.NewFileSet()
	imp := integrationImporter(t, fset)
	tests := map[string]struct {
		callback string
		want     []string
	}{
		"error: call through a function and a method":      {"indirect", []string{root + ": GET /sample -> indirect -> (runner).invoke -> llama.CallRaw"}},
		"success: unrelated helpers and same-named method": {"func() { clean{}.invoke() }", nil},
		"success: local closure":                           {"func() { fn := func() { clean{}.invoke() }; fn() }", nil},
		"error: interface dispatch":                        {"func() { dynamic.invoke() }", []string{root + ": GET /sample -> unresolved call dynamic.invoke()"}},
		"error: function variable assigned in init":        {"func() { later() }", []string{root + ": GET /sample -> unresolved call later()"}},
		"error: function variable as callback":             {"later", []string{root + ": GET /sample -> unresolved callback later"}},
		"error: reassigned local closure":                  {"func() { fn := func() {}; fn = later; fn() }", []string{root + ": GET /sample -> unresolved call fn()"}},
		"error: function field assigned in init":           {"func() { holder.run() }", []string{root + ": GET /sample -> unresolved call holder.run()"}},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var syntax []*ast.File
			for path, src := range map[string]string{root: "package integration\ntype contractCase struct{call func()}\nfunc registerContracts(c map[string]contractCase) bool {return true}\nvar _ = registerContracts(map[string]contractCase{\"GET /sample\": {call:" + tt.callback + "}})", "helpers.go": helpers} {
				f, err := parser.ParseFile(fset, path, src, parser.SkipObjectResolution)
				if err != nil {
					t.Fatal(err)
				}
				syntax = append(syntax, f)
			}
			if diff := gocmp.Diff(tt.want, contractBridgeChains(t, fset, syntax, []string{root}, imp)); diff != "" {
				t.Errorf("reachable bridges (-want +got):\n%s", diff)
			}
		})
	}
}

// contractFiles lists the contract_<name>_test.go files of dir, without contract_test.go itself, which holds the
// test machinery and no cases.
func contractFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range entries {
		name := e.Name()
		if e.Type().IsRegular() && name != "contract_test.go" && strings.HasPrefix(name, "contract_") && strings.HasSuffix(name, "_test.go") {
			files = append(files, name)
		}
	}
	return files, nil
}

// unlistedRegistrations reports registrations made from a file that is not among the parsed contract files, which
// would escape the check for typed methods.
func unlistedRegistrations(regs []contractRegistration, files []string) []string {
	var problems []string
	for _, reg := range regs {
		if !slices.Contains(files, reg.file) {
			problems = append(problems, fmt.Sprintf("contract cases registered from %s, which is not a contract_<name>_test.go file of the package directory; move them into one", reg.file))
		}
	}
	return problems
}

func TestMergeContracts(t *testing.T) {
	health := contractCase{server: presetStories260K}
	tests := map[string]struct {
		regs         []contractRegistration
		wantKeys     []string
		wantProblems []string
	}{
		"success: disjoint files": {
			regs: []contractRegistration{
				{file: "contract_a_test.go", cases: map[string]contractCase{"GET /health": health}},
				{file: "contract_b_test.go", cases: map[string]contractCase{"GET /props": health}},
			},
			wantKeys: []string{"GET /health", "GET /props"},
		},
		"error: one key in two files": {
			regs: []contractRegistration{
				{file: "contract_a_test.go", cases: map[string]contractCase{"GET /health": health}},
				{file: "contract_b_test.go", cases: map[string]contractCase{"GET /health": health, "GET /props": health}},
			},
			wantKeys:     []string{"GET /health", "GET /props"},
			wantProblems: []string{`endpoint "GET /health" has contract cases in both contract_a_test.go and contract_b_test.go`},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			all, problems := mergeContracts(tt.regs)
			if got := slices.Sorted(maps.Keys(all)); !slices.Equal(got, tt.wantKeys) {
				t.Errorf("keys = %v, want %v", got, tt.wantKeys)
			}
			if !slices.Equal(problems, tt.wantProblems) {
				t.Errorf("problems = %q, want %q", problems, tt.wantProblems)
			}
		})
	}
}

func TestUnlistedRegistrations(t *testing.T) {
	regs := []contractRegistration{
		{file: "contract_health_test.go"},
		{file: "stream_test.go"},
	}
	got := unlistedRegistrations(regs, []string{"contract_health_test.go"})
	if len(got) != 1 || !strings.Contains(got[0], "stream_test.go") {
		t.Errorf("unlistedRegistrations = %q, want one problem naming stream_test.go", got)
	}
	if got := unlistedRegistrations(regs[:1], []string{"contract_health_test.go"}); len(got) != 0 {
		t.Errorf("a listed file was reported: %q", got)
	}
}

// TestRegisterContractsRecordsTheCaller checks that the file comes from the call site.
func TestRegisterContractsRecordsTheCaller(t *testing.T) {
	saved := contractRegistrations
	t.Cleanup(func() { contractRegistrations = saved })
	contractRegistrations = nil
	registerContracts(map[string]contractCase{})
	if len(contractRegistrations) != 1 || contractRegistrations[0].file != "contract_test.go" {
		t.Errorf("registration = %+v, want one from contract_test.go", contractRegistrations)
	}
}

func TestContractFilesListing(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"contract_test.go", "contract_health_test.go", "contract_x_test.go", "contractx_test.go", "other_test.go", "contract_y.go"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("package p\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := contractFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"contract_health_test.go", "contract_x_test.go"}; !slices.Equal(got, want) {
		t.Errorf("contractFiles = %v, want %v", got, want)
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

// TestRouteMatchesQueryExactly checks that a discriminator must be sent alone: the server keeps one value of a
// repeated parameter, so "action=save&action=erase" would erase.
func TestRouteMatchesQueryExactly(t *testing.T) {
	r, err := parseEndpointKey("POST /slots/{id_slot}?action=save")
	if err != nil {
		t.Fatal(err)
	}
	tests := map[string]struct {
		query url.Values
		extra url.Values
		want  bool
	}{
		"match: the discriminator alone":         {query: url.Values{"action": {"save"}}, want: true},
		"match: other parameters beside it":      {query: url.Values{"action": {"save"}, "x": {"1"}}, want: true},
		"no match: repeated with another value":  {query: url.Values{"action": {"save", "erase"}}, want: false},
		"no match: repeated with the same value": {query: url.Values{"action": {"save", "save"}}, want: false},
		"no match: other value":                  {query: url.Values{"action": {"erase"}}, want: false},
		"no match: missing":                      {query: url.Values{}, want: false},
		"match: required extra parameter":        {query: url.Values{"action": {"save"}, "model": {"m"}}, extra: url.Values{"model": {"m"}}, want: true},
		"no match: extra parameter among others": {query: url.Values{"action": {"save"}, "model": {"a", "m"}}, extra: url.Values{"model": {"m"}}, want: false},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			ex := harness.Exchange{Method: http.MethodPost, Path: "/slots/0", Query: tt.query, Header: http.Header{}}
			if got := r.matches(ex, tt.extra); got != tt.want {
				t.Errorf("matches(%v, extra %v) = %t, want %t", tt.query, tt.extra, got, tt.want)
			}
		})
	}
}
