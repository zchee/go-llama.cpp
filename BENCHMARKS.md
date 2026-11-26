# Benchmarks

**The recorded chat stream misses the budget.** Decoding nine payloads takes 115 allocations, down from 124 on the base commit. Bare `encoding/json/v2.Unmarshal` into the same reusable `ChatCompletionChunk` takes 81. The overhead is **(115 − 81) / 9 = 3.78 allocations per event**; the target is at most 3, or 108 allocations per stream. The `[DONE]` frame is parsed but not yielded or counted as a payload.

- Run started (`date`): `2026-11-27 03:36:01 JST`
- Base (`git rev-parse HEAD`): `ba3e27e21876d8010c02c1bf9f706f63d25dd99c`, with the decoder and fresh-pointer benchmark changes uncommitted.
- Host: Apple M3 Max, `go version go1.27.4 darwin/arm64`.
- Command: `go test -run '^$' -bench . -benchmem -count=10 <pkg>` for `.`, `./openai`, and `./anthropic`, one package at a time. Tables show benchstat medians; `KiB` means 1024 bytes. Other worktrees ran tests on the same machine, so timings are noisy. Allocation and event counts were identical across all ten samples.

## Recorded streams

| Decoder | Yielded events | Base allocs/stream | New allocs/stream | Allocations/event, base → new |
| --- | ---: | ---: | ---: | ---: |
| Native completion | 5 | 124 | 119 | 24.80 → 23.80 |
| Single chat token chunk | 1 | 28 | 27 | 28.00 → 27.00 |
| Whole chat completion | 9 | 124 | 115 | 13.78 → 12.78 |
| Responses | 14 | 139 | 125 | 9.93 → 8.93 |
| Messages | 13 | 101 | 88 | 7.77 → 6.77 |

The original chat baseline reuses one destination. The stream must return a distinct pointer for each event so the caller can keep it. A second benchmark unmarshals the same nine payloads into fresh pointers: 90 allocations rather than 81. The decoder costs 25 extra allocations against that comparator (2.78/event), but **the 81-allocation benchmark remains the budget baseline**. For the single token chunk, the stream takes 27 allocations versus 17 for bare unmarshal; it pays the entire setup and terminator cost for one payload.

### What still allocates

Base and final profiles used `-bench '^BenchmarkChatCompletionStreamDecode$' -benchtime=1000x -memprofilerate=1 -memprofile` on `./openai`, inspected with `go tool pprof -sample_index=alloc_objects -top -lines`. The two base `readStream` sites each accounted for 9,000 objects: `[]byte(ev.Data)` and `new(T)`. Only `new(T)` remains. `json.UnmarshalRead` uses a resettable `strings.Reader` for ordinary payloads; it still validates one complete JSON value. The event string is not exposed as mutable bytes.

| Source of overhead vs. bare unmarshal | Allocations per nine-event stream |
| --- | ---: |
| Fresh result pointers (`new(T)`) | 9 |
| SSE scanner token strings and data builder buffers for the nine payloads | 18 |
| Same SSE scanner and builder for `[DONE]` | 2 |
| SSE parser, field parser, `bufio.Scanner` and its initial buffer | 4 |
| Shared allocation for the stream-tail reader and JSON payload reader | 1 |
| **Total** | **34** |

In `go-sse` v0.12.0, `internal/parser/parser.go:79` calls `Scanner.Text()` **once per framed event**, not once per line. `event.go:85,114` writes the payload and resets the builder for the next event. `ReadConfig` accepts a size limit but no parser or buffer; `Connection` uses the same internal parser and builder. The nine sets of scanner, builder and result allocations already consume the 27-object allowance before setup or `[DONE]`. Even a zero-copy string conversion would not remove those costs. No parser was replaced, and the limit and budget are unchanged.

## All benchmarks

### Root package

| Benchmark | Time/op | Bytes/op | Allocs/op |
| --- | ---: | ---: | ---: |
| `RequestWithoutExtra/checkRequestExtra` | 154.2 ns ± 31% | 0 | 0 |
| `RequestWithoutExtra/marshal` | 850.8 ns ± 4% | 128 | 4 |
| `RequestWithoutExtra/encodeBody` | 1.035 µs ± 89% | 128 | 4 |
| `CompletionStreamDecode` | 25.76 µs ± 4% | 16.14 KiB | 119 |
| `PromptEncode/text` | 76.83 ns ± 2% | 8 | 1 |
| `PromptEncode/tokens` | 150.5 ns ± 2% | 56 | 3 |
| `PromptEncode/mixed` | 102.5 ns ± 2% | 16 | 1 |
| `PromptEncode/object` | 270.1 ns ± 2% | 128 | 2 |
| `PromptEncode/batch` | 103.4 ns ± 2% | 16 | 1 |
| `ClientNewRequest/get` | 859.0 ns ± 12% | 1.570 KiB | 16 |
| `ClientNewRequest/post_json` | 2.265 µs ± 3% | 1.900 KiB | 24 |

### OpenAI package

| Benchmark | Time/op | Bytes/op | Allocs/op |
| --- | ---: | ---: | ---: |
| `ChatCompletionChunkUnmarshal` | 2.364 µs ± 4% | 656 | 17 |
| `ChatCompletionChunkStream` | 4.707 µs ± 32% | 6.067 KiB | 27 |
| `ChatCompletionStreamDecode` | 23.49 µs ± 3% | 16.56 KiB | 115 |
| `ChatCompletionStreamUnmarshalBaseline` | 14.96 µs ± 2% | 3.534 KiB | 81 |
| `ChatCompletionStreamUnmarshalFreshPointerBaseline` | 15.48 µs ± 6% | 4.942 KiB | 90 |
| `ResponsesStreamDecode` | 50.21 µs ± 3% | 18.26 KiB | 125 |
| `EmbeddingResponseDecode/float` | 43.73 µs ± 1% | 16.15 KiB | 13 |
| `EmbeddingResponseDecode/base64` | 2.950 µs ± 4% | 4.652 KiB | 5 |
| `ChatCompletionRequestWithToolsEncode` | 2.242 µs ± 1% | 744 | 9 |

### Anthropic package

| Benchmark | Time/op | Bytes/op | Allocs/op |
| --- | ---: | ---: | ---: |
| `MessagesStreamDecode` | 11.82 µs ± 22% | 9.419 KiB | 88 |
