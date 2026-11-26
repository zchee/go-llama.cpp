# Benchmarks

- Date: `2026-11-27 02:59:05 JST`
- Commit: `284b291b492df973209d8df77ff06bc4df63b32c`
- CPU: `Apple M3 Max` (`darwin/arm64`)
- Go: `go version go1.27.4 darwin/arm64`
- Command: `go test -run '^$' -bench . -benchmem -count=10 <pkg>` (each package run sequentially)
- Summarized with: `benchstat`

Note: Concurrent test runs on other worktrees loaded this machine while these benchmarks ran, so latency figures (`sec/op`) are indicative. Allocation counts (`B/op`, `allocs/op`) and event counts (`events/op`) are deterministic.

## Results

### Root package (`github.com/zchee/go-llama.cpp`)

| Benchmark | sec/op | events/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| `CompletionStreamDecode-16` | `25.92µ ± 2%` | `5.000 ± 0%` | `19.86Ki ± 0%` | `124.0 ± 0%` |
| `PromptEncode/text-16` | `93.16n ± 101%` | — | `8.000 ± 0%` | `1.000 ± 0%` |
| `PromptEncode/tokens-16` | `156.7n ± 57%` | — | `56.00 ± 0%` | `3.000 ± 0%` |
| `PromptEncode/mixed-16` | `108.8n ± 12%` | — | `16.00 ± 0%` | `1.000 ± 0%` |
| `PromptEncode/object-16` | `202.7n ± 2%` | — | `64.00 ± 0%` | `1.000 ± 0%` |
| `PromptEncode/batch-16` | `104.3n ± 2%` | — | `16.00 ± 0%` | `1.000 ± 0%` |
| `ClientNewRequest/get-16` | `770.4n ± 20%` | — | `1.570Ki ± 0%` | `16.00 ± 0%` |
| `ClientNewRequest/post_json-16` | `2.064µ ± 2%` | — | `1.900Ki ± 0%` | `24.00 ± 0%` |

### OpenAI package (`github.com/zchee/go-llama.cpp/openai`)

| Benchmark | sec/op | events/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| `ChatCompletionChunkUnmarshal-16` | `2.814µ ± 13%` | — | `656.0 ± 0%` | `17.00 ± 0%` |
| `ChatCompletionChunkStream-16` | `3.975µ ± 9%` | `1.000 ± 0%` | `6.505Ki ± 0%` | `28.00 ± 0%` |
| `ChatCompletionStreamDecode-16` | `24.18µ ± 2%` | `9.000 ± 0%` | `20.10Ki ± 0%` | `124.0 ± 0%` |
| `ChatCompletionStreamUnmarshalBaseline-16` | `15.00µ ± 2%` | `9.000 ± 0%` | `3.534Ki ± 0%` | `81.00 ± 0%` |
| `ResponsesStreamDecode-16` | `52.19µ ± 1%` | `14.00 ± 0%` | `22.77Ki ± 0%` | `139.0 ± 0%` |
| `EmbeddingResponseDecode/float-16` | `45.39µ ± 11%` | — | `16.15Ki ± 0%` | `13.00 ± 0%` |
| `EmbeddingResponseDecode/base64-16` | `3.032µ ± 4%` | — | `4.652Ki ± 0%` | `5.000 ± 0%` |
| `ChatCompletionRequestWithToolsEncode-16` | `2.269µ ± 2%` | — | `744.0 ± 0%` | `9.000 ± 0%` |

### Anthropic package (`github.com/zchee/go-llama.cpp/anthropic`)

| Benchmark | sec/op | events/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| `MessagesStreamDecode-16` | `11.76µ ± 15%` | `13.00 ± 0%` | `10.70Ki ± 0%` | `101.0 ± 0%` |

## Stream Allocation Comparison vs. Baseline

Target: decoding a chat stream through `llama.DecodeStream` costs at most `3` allocations per event more than bare `json.Unmarshal` (`encoding/json/v2`) of the same payload.

- **Measured result (missed target):**
  - Whole recorded chat stream (`openai/testdata/chat_completions/stream.resp.sse`, 9 events):
    - `ChatCompletionStreamDecode`: `124.0` allocs/op (`13.78` allocs/event)
    - `ChatCompletionStreamUnmarshalBaseline`: `81.00` allocs/op (`9.00` allocs/event)
    - Overhead: `+43.0` allocs/op = **`+4.78` extra allocs/event** (misses the `<= 3` extra allocs/event target by `+1.78` allocs/event).
  - Single chat token chunk (event index 3 from `stream.resp.sse`):
    - `ChatCompletionChunkStream`: `28.00` allocs/op
    - `ChatCompletionChunkUnmarshal`: `17.00` allocs/op
    - Overhead: **`+11.0` allocs/op** (includes per-stream setup and the `[DONE]` terminator frame).

### Allocation Sources (`-memprofile`)

Profiling `BenchmarkChatCompletionStreamDecode` with `go test -memprofile` (`47,184` iterations, `5,414,312` total allocated objects; `go tool pprof -alloc_objects`) attributes the stream-layer overhead to:

1. **Per-event allocations (4 allocs/event):**
   - `github.com/tmaxmax/go-sse/internal/parser.(*Parser).Next` (`parser.go:79`): `r.fieldScanner.Reset(r.inputScanner.Text())` allocates a `string` via `bufio.(*Scanner).Text` for each line.
   - `github.com/tmaxmax/go-sse.read.func1` (`event.go:85` + `event.go:114`): `sb.Reset()` discards the `strings.Builder` buffer after each event, so `sb.WriteString(f.Value)` allocates a fresh buffer per event.
   - `github.com/zchee/go-llama.cpp.readStream` (`stream.go:119`): `data := []byte(ev.Data)` allocates a `[]byte` copy of the event string for `json.Unmarshal`.
   - `github.com/zchee/go-llama.cpp.readStream` (`stream.go:124`): `v := new(T)` allocates each yielded `*openai.ChatCompletionChunk` on the heap (whereas the bare `json.Unmarshal` baseline unmarshals into a stack-local struct).
2. **Per-stream fixed and terminator allocations (~7 allocs/stream, ~0.78 allocs/event over 9 events):**
   - `stream.go:103`: `tr := &tailReader{r: r}` (1 alloc/stream).
   - `github.com/tmaxmax/go-sse/internal/parser.New`: `bufio.NewScanner`, initial `bufio.Scanner` buffer in `Scan()`, and `parser.NewFieldParser` (3 allocs/stream).
   - `[DONE]` terminator event: `bufio.(*Scanner).Text` (`parser.go:79`) and `strings.(*Builder).WriteString` (`event.go:85`) before `ev.Data == streamDone` returns (2 allocs/stream, plus minor `runtime` / closure overhead).
