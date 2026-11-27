# go-llama.cpp

A Go client for the HTTP API of [llama.cpp's server](https://github.com/ggml-org/llama.cpp/tree/v1.1.0/tools/server) (`llama-server`) at tag v1.1.0. See the [table of 38 documented endpoints](#endpoints) for the corresponding Go methods.

This is not a binding to the C library. Run `llama-server` separately and pass its base URL to `llama.New`. The client doesn't start, stop or supervise the server process.

The root package `llama` covers the native endpoints, including router model management. The `openai` and `anthropic` packages handle their respective compatibility endpoints, `systemone` evaluates structured questions with a TypeSafe-compatible decision model, and `realtime` opens WebSocket sessions. Each uses a `*llama.Client` for its connection settings.

The wire types follow the [server implementation where it differs from its README](#where-the-server-differs-from-its-readme).

## Requirements

- Go 1.27 or newer. The wire types are built on `encoding/json/v2` and `encoding/json/jsontext`.
- A `llama-server` to talk to; the integration tests pin version 1.1.0.

## Install

```sh
go get github.com/zchee/go-llama.cpp
```

## Quick start

The examples use a server at `localhost:8080`. For the native completion example, replace `model.gguf` with a local text model:

```sh
llama-server --model model.gguf --host 127.0.0.1 --port 8080
```

The Go snippets come from the package examples. To run one as a program, use `package main`, rename its function to `main`, and import `context`, `fmt`, and `llama "github.com/zchee/go-llama.cpp"`. The OpenAI, Anthropic, realtime, and System One snippets also import the corresponding subpackage. The Anthropic and realtime snippets use `log`.

`llama.New` validates the base URL and options. Call `WaitReady` with a context to poll `/health` while the server loads the model. Both `Health` and `WaitReady` require a JSON response with `status: "ok"`; an HTTP success status alone is not enough.

### Native completion, streamed

`CompletionStream` returns an `iter.Seq2[*CompletionResponse, error]`. Nothing is sent until the loop starts, and the response body is closed when the loop ends, also on break and context cancellation.

```go
func ExampleClient_CompletionStream() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	req := &llama.CompletionRequest{
		Prompt:   llama.TextPrompt("Once upon a time"),
		NPredict: new(16),
	}
	for chunk, err := range c.CompletionStream(context.Background(), req) {
		if err != nil {
			panic(err)
		}
		if chunk.Content != nil {
			fmt.Print(*chunk.Content)
		}
	}
}
```

### OpenAI chat completion with a tool

Start `llama-server` with `--jinja` and use a chat template that supports tools. `llama.SchemaFor` builds the tool's argument schema from a Go type. The example prints the requested function and arguments; the client does not execute it.

```go
func ExampleClient_ChatCompletions_tool() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	type weatherArgs struct {
		City string `json:"city"`
	}
	params, err := llama.SchemaFor[weatherArgs]()
	if err != nil {
		panic(err)
	}
	res, err := openai.NewClient(c).ChatCompletions(context.Background(), &openai.ChatCompletionRequest{
		Messages: []openai.Message{{Role: "user", Content: openai.TextContent("What is the weather in Kyoto?")}},
		Tools: []openai.Tool{{Type: "function", Function: &openai.FunctionDefinition{
			Name:        "weather",
			Description: new("Get the weather for a city."),
			Parameters:  params,
		}}},
		ToolChoice: openai.ToolChoiceFunction("weather"),
		MaxTokens:  new(128),
	})
	if err != nil {
		panic(err)
	}
	for i := range res.Choices {
		for _, call := range res.Choices[i].Message.ToolCalls {
			if f := call.Function; f != nil && f.Name != nil && f.Arguments != nil {
				fmt.Printf("%s: %s\n", *f.Name, *f.Arguments)
			}
		}
	}
}
```

### Embeddings

Start an embedding model with `--embedding` and a pooling mode other than `none` for `/v1/embeddings`. Set `Input` on `openai.EmbeddingRequest`; for legacy `content` requests, use the native `Embedding` or `Embeddings` method. The example uses float arrays. Base64 decoding assumes a little-endian server; use float arrays if its byte order is unknown.

```go
func ExampleClient_Embeddings_float() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	resp, err := openai.NewClient(c).Embeddings(context.Background(), &openai.EmbeddingRequest{
		Input: llama.EmbeddingTexts("first sentence", "second sentence"),
	})
	if err != nil {
		panic(err)
	}
	for i := range resp.Data {
		fmt.Println(resp.Data[i].Index, resp.Data[i].Embedding.Len())
	}
}
```

### Anthropic messages

```go
func ExampleClient_Messages() {
	c, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		log.Fatal(err)
	}
	res, err := anthropic.NewClient(c).Messages(context.Background(), &anthropic.MessagesRequest{Messages: []anthropic.Message{{Role: "user", Content: anthropic.TextContent("Tell a short story.")}}, MaxTokens: new(64)})
	if err != nil {
		log.Fatal(err)
	}
	for i := range res.Content {
		if text := res.Content[i].Text; text != nil {
			fmt.Print(*text)
		}
	}
}
```

### Realtime session

Use a model that supports turn-based realtime conversation. `realtime.Dial` opens the WebSocket with the HTTP client and authentication settings of the `*llama.Client`. Its context bounds only the handshake; close the session when you're done.

```go
func ExampleSession() {
	c, err := llama.New("http://127.0.0.1:8080")
	if err != nil {
		log.Print(err)
		return
	}
	ctx := context.Background()
	s, err := realtime.Dial(ctx, c, nil)
	if err != nil {
		log.Print(err)
		return
	}
	defer func() {
		if err := s.CloseNow(); err != nil {
			log.Print(err)
		}
	}()
	if err := s.CreateItem(ctx, realtime.Item{Type: "message", Role: new("user"), Content: []realtime.ContentPart{{Type: "input_text", Text: new("Hello")}}}); err != nil {
		log.Print(err)
		return
	}
	if err := s.CreateResponse(ctx, &realtime.ResponseConfig{OutputModalities: realtime.Nullable[[]string]{Value: new([]string{"text"})}}); err != nil {
		log.Print(err)
		return
	}
	for ev, err := range s.Events() {
		if err != nil {
			log.Print(err)
			return
		}
		switch ev.Type {
		case realtime.ResponseOutputTextDelta:
			fmt.Print(*ev.Delta)
		case realtime.ErrorEvent:
			log.Print(ev.Error.Message)
			return
		case realtime.ResponseDone:
			return
		}
	}
}
```

The integration suite exercises live transcription and turn-based conversation. Full-duplex conversation has not been verified end to end because the available test models do not support it. Set `LLAMA_TEST_FULL_DUPLEX_MODEL` (and optionally `LLAMA_TEST_FULL_DUPLEX_MMPROJ`) to run the opt-in full-duplex test against a duplex-capable GGUF.

### System One decision

Use a System One decision model. Set `systemone.Request.Model` when using the router to select the model; a single-model server ignores it. Use `systemone.Raw` to pass encoded JSON, or `systemone.Text` for text. Supported payload kinds depend on the model. `Payload.Kind()`, `Text()`, and `IsNull()` inspect the value; `Payload.Raw()` copies encoded objects, arrays, numbers, or booleans.

```go
func ExampleClient_Evaluate() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	res, err := systemone.NewClient(c).Evaluate(context.Background(), &systemone.Request{
		State: systemone.Text("My parcel was delivered."),
		Questions: map[string]systemone.Question{
			"delivered": systemone.Noul(systemone.Text("Has the parcel arrived?"), nil),
		},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Answers["delivered"].Value.(systemone.NoulAnswer).Noul)
}
```

### Router model management

Start the server in router mode, for example with `--models-dir ./models` instead of `--model`. Replace `qwen2.5-0.5b-instruct` with an ID from `ListModels`. `LoadModel` asks the router to start loading; a successful call does not mean the model is ready yet.

```go
func ExampleClient_LoadModel() {
	c, err := llama.New("http://localhost:8080")
	if err != nil {
		panic(err)
	}
	ctx := context.Background()
	if _, err := c.LoadModel(ctx, &llama.ModelRequest{Model: "qwen2.5-0.5b-instruct"}); err != nil {
		panic(err)
	}
	list, err := c.ListModels(ctx)
	if err != nil {
		panic(err)
	}
	for i := range list.Data {
		m := &list.Data[i]
		if m.Status != nil {
			fmt.Println(m.ID, m.Status.Value)
		}
	}
}
```

`ModelEvents` streams model status and download progress. `DownloadModel` and `DeleteModel` manage the router's cache; `DeleteModel` requires a non-empty model name.

## Endpoints

Every endpoint the server README documents at v1.1.0, and the typed method that reaches it. Use `llama.Call`, `llama.Stream`, or `llama.CallRaw` for aliases (such as `/chat/completions`) and routes without a typed method. The client copies raw request bytes supplied as `llama.Body` or `jsontext.Value` when it builds the request.

| Endpoint | Method |
|---|---|
| `GET /health` | `(*llama.Client).Health`; `WaitReady` polls it |
| `POST /completion` | `(*llama.Client).Completion`, `CompletionBatch`, `CompletionStream` |
| `POST /tokenize` | `(*llama.Client).Tokenize` |
| `POST /detokenize` | `(*llama.Client).Detokenize` |
| `POST /apply-template` | `(*llama.Client).ApplyTemplate` |
| `POST /embedding` | `(*llama.Client).Embedding` |
| `POST /reranking` | `(*llama.Client).Rerank` (Jina form), `RerankTEI` (TEI form) |
| `POST /tts` | `(*llama.Client).TTS` |
| `POST /infill` | `(*llama.Client).Infill`, `InfillBatch`, `InfillStream` |
| `GET /props` | `(*llama.Client).Props` |
| `POST /props` | `(*llama.Client).SetProps` |
| `POST /embeddings` | `(*llama.Client).Embeddings` |
| `GET /slots` | `(*llama.Client).Slots` |
| `GET /metrics` | `(*llama.Client).Metrics`, `MetricsText` |
| `POST /slots/{id_slot}?action=save` | `(*llama.Client).SaveSlot` |
| `POST /slots/{id_slot}?action=restore` | `(*llama.Client).RestoreSlot` |
| `POST /slots/{id_slot}?action=erase` | `(*llama.Client).EraseSlot` |
| `GET /lora-adapters` | `(*llama.Client).LoRAAdapters` |
| `POST /lora-adapters` | `(*llama.Client).SetLoRAAdapters` |
| `GET /v1/models` | `(*openai.Client).Models` |
| `POST /v1/completions` | `(*openai.Client).Completions`, `CompletionsStream` |
| `POST /v1/chat/completions` | `(*openai.Client).ChatCompletions`, `ChatCompletionsStream` |
| `POST /v1/chat/completions/control` | `(*openai.Client).ControlChatCompletion` |
| `WebSocket /v1/realtime` | `realtime.Dial`, then the `realtime.Session` methods |
| `POST /v1/audio/speech` | `(*openai.Client).Speech` |
| `POST /v1/responses` | `(*openai.Client).Responses`, `ResponsesStream` |
| `POST /v1/embeddings` | `(*openai.Client).Embeddings` |
| `POST /v1/responses/input_tokens` | `(*openai.Client).ResponsesInputTokens` |
| `POST /v1/chat/completions/input_tokens` | `(*openai.Client).ChatInputTokens` |
| `POST /v1/messages` | `(*anthropic.Client).Messages`, `MessagesStream` |
| `POST /v1/messages/count_tokens` | `(*anthropic.Client).CountTokens` |
| `POST /v1/systemone` | `(*systemone.Client).Evaluate` |
| `GET /models` | `(*llama.Client).ListModels` |
| `POST /models/load` | `(*llama.Client).LoadModel` |
| `POST /models/unload` | `(*llama.Client).UnloadModel` |
| `GET /models/sse` | `(*llama.Client).ModelEvents` |
| `POST /models` | `(*llama.Client).DownloadModel` |
| `DELETE /models` | `(*llama.Client).DeleteModel` |

The integration suite checks the server's refusal responses to `reasoning_end`. It has not verified that the action ends reasoning on an active completion; that requires a model and template with reasoning control enabled.

## Client options

`llama.New(baseURL, opts...)` takes:

- `WithAPIKey(key)` — the key the server was started with (`--api-key`). Sent as `Authorization: Bearer` by default; `WithAPIKeyHeader(llama.APIKeyHeaderXAPIKey)` switches to `X-Api-Key`. The server accepts both.
- `WithUnixSocket(path)` — connect to a server listening on a Unix domain socket (`--host` ending in `.sock`). The base URL still supplies the scheme, Host header and API prefix; `http://localhost` is usually enough. The effective transport must be a non-nil `*http.Transport` so the client can clone it. Without a Unix socket, a wrapped default transport is supported.
- `WithRetry(policy)` — replace the default of up to four attempts with exponential backoff. The client retries HTTP 503 with `unavailable_error`, or a transport failure before a response for an idempotent request. It does not retry an accepted stream or a redirect refused by the HTTP client. The client accepts `Retry-After` as decimal seconds or an HTTP date and caps the wait at the policy’s maximum interval. Use `WithNoRetry()` to limit the library to one attempt for a request, or `llama.RetryPolicy{MaxAttempts: 1}` for the whole client. Neither changes `net/http` redirect and replay behavior.
- `WithLogger(l)` — log request and retry details at debug level using `slog`. The client does not log request bodies or the API key.
- `WithHTTPClient(hc)` — supply the HTTP client, including for WebSocket handshakes. An overall HTTP timeout can interrupt an active HTTP stream; prefer a context deadline for each request.
- `WithUserAgent(ua)` and `WithMaxEventSize(n)` — set the user agent and maximum size of an SSE event.

Request options include `WithQuery`, `WithHeader`, and `WithAutoload`. For model selection, use `WithModelQuery` for GET requests, `DialOptions.Model` for WebSocket handshakes, and the `model` body field for POST requests routed to a model. Pass handshake request options through `realtime.DialOptions.RequestOptions`.

`WithFailOnNoSlot()` sets `fail_on_no_slot=1` and limits the library to one attempt. Setting the same query in the path or with `WithQuery` returns the `/slots` 503 `no slot available` without retrying, but still allows retries if the model is loading or the connection fails.

## Errors

Methods that require a request reject `nil` before sending, for example `llama: nil completion request`. A streaming method yields that error when iteration starts.

Non-2xx HTTP responses return an `*llama.APIError`. Use `errors.AsType[*llama.APIError](err)` to inspect its `StatusCode` and `Type`; for example, `llama.ErrorTypeExceedContextSize` tells you to shorten the prompt. An SSE error event also yields an `*llama.APIError` and ends the iterator. In both cases `APIError.Raw` contains at most 1 MiB of the original payload. The client redacts the configured API key when formatting returned errors, including with `%#v`. Realtime `error` events are delivered as `ServerEvent` values, so a caller can handle the event and keep the session open.

An SSE stream that ends before its required final frame yields an error that matches `errors.Is(err, llama.ErrIncompleteStream)`. Native completion and infill expect final result frames; OpenAI text and chat completions expect `[DONE]`; Responses expects `response.completed`; Anthropic Messages expects `message_stop`. This check confirms that the end arrived, not that no middle event was lost. `ModelEvents` has no end marker, so EOF at a frame boundary is normal. Stopping the iteration early yourself is not an incomplete-stream error. To apply the endpoint checks when replaying a recording, use `llama.DecodeEndpointStream`; `llama.DecodeStream` checks SSE framing only.

## Resource ownership

- **HTTP server-sent event streams are lazy and single-use.** The `iter.Seq2` sends the request when iteration starts. The client closes the response body on completion, error (including a missing final frame), `break`, or context cancellation. A second iteration yields `llama.ErrStreamConsumed` without sending another request.
- **The caller owns audio bodies.** Close `Audio.Body` after `TTS` or `Speech`. `Audio.Format` describes the advertised encoding. `Audio.WriteTo` copies the audio and closes the body, stopping without draining it if a write fails. A normal EOF on an audio stream does not prove speech generation succeeded: the server can end it the same way after a generation error.
- **The caller owns realtime sessions.** Use `Close(ctx)` to wait for the closing handshake or `CloseNow()` to close without waiting for it, even if a closing handshake is already in progress. Breaking out of `Events()` leaves the connection open. Only one `Events()` iterator may read at a time, and writes can run concurrently with it.

## Unknown members survive a round trip

Wire object types with `Extra map[string]jsontext.Value` (tagged `json:",embed"`) keep unrecognized JSON members so they survive decoding and re-encoding. Before sending a typed request, including a multipart speech request or a typed realtime client event, the client rejects any `Extra` key that exactly matches a typed JSON member of the same object, even if the typed field is unset. This also applies to objects in a top-level slice, array, or map. Use the typed field instead, for example `systemone.Request.Model` rather than `Extra["model"]`.

Unrecognized tool-choice forms are available through `ToolChoice.Unknown()`, and unknown realtime frames through `ClientEvent.Raw` or `ServerEvent.Raw`. Realtime tool definitions carry `DeferLoading` and `Strict` as optional booleans. `anthropic.RawContent` and `systemone.Raw` accept encoded JSON for content and payloads, respectively.

### Wire types

- Optional request `Model` fields are strings. Leave them empty to omit `model`; required model names still need a value.
- Optional pointer fields distinguish an omitted value from an explicit zero or empty string. `TTSParams.Input` is a `*string`: omit it to use the `Prompt` alias. If a speech request sets `RawPrompt` with a speaker reference file, the client sends JSON with the reference bytes instead of multipart so the server reads the boolean correctly. Use `NamedVoice` or `ReferencedVoice` in `openai` and `realtime` to select a voice by name or ID.
- `PromptObject.PromptString` and `EmbeddingPromptObject.PromptString` use `TokenizeContent`: `TokenizeText` for text or `TokenizeParts` for a flat list without attached media. The multimodal object form requires text when media is present.
- Build Responses input and content with `openai.ResponseInputText`, `ResponseInputItems`, `ResponseContentText`, or `ResponseContentParts`; read the selected form with `Text()`, `Items()`, or `Parts()`. Slice accessors on values with multiple wire forms return read-only views. Clone a slice before changing its elements.
- `llama.Continuation` controls `continue_final_message` for both template and chat requests; `openai.ChatContinuation` is an alias. For OpenAI text completions, `NCmpl` takes precedence over `N`, and the server checks `NPredict`, `MaxCompletionTokens`, then `MaxTokens` in that order. The DRY repetition penalty fields on `SamplingParams` and `GenerationSettings` are `DRYMultiplier`, `DRYBase`, `DRYAllowedLength`, `DRYPenaltyLastN`, and `DRYSequenceBreakers`.
- `RerankRequest.TopN` and `RerankTEIRequest.TopN` are `*int32`: nil returns all results, zero returns none. `RerankObject.ClearText` and `ClearImage` omit those members, including after decoding an explicit null. `PropsResponse.UISettings` holds the raw JSON as `jsontext.Value`. Slot save, restore, and erase reject negative slot IDs before sending.

## This library or an official SDK?

If your application also talks to OpenAI or Anthropic, use the corresponding official SDK ([openai-go](https://github.com/openai/openai-go) or [anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go)) and point its base URL at `llama-server` for supported operations. Use this library for native routes such as `/infill` or `/slots`, llama.cpp-specific fields such as `id_slot`, or the typed System One decision endpoint, which neither official SDK covers. The compatibility routes do not implement every feature of the hosted APIs.

## Testing

- Unit tests run without a server. They include decoding fixtures recorded from a real `llama-server` 1.1.0 (`go test ./...`).
- Integration tests run real servers and are enabled with `LLAMA_INTEGRATION=1 go test ./integration/...`. The binary comes from `LLAMA_SERVER_BIN` or `llama-server` on `PATH`; the tests target version 1.1.0. Models are pinned by revision and SHA-256 in `integration/models.yaml` and cached locally. The test harness can pass additional environment variables to a server via `harness.Options.Env`.
- Twelve integration cases compare supported OpenAI and Anthropic wire formats with the official Go SDKs using responses from the real server. Only the integration module depends on the official SDKs. The tests use no mock servers.

The recorded chat stream decoder misses its allocation budget relative to bare JSON unmarshalling; [BENCHMARKS.md](BENCHMARKS.md) shows by how much and why.

## Where the server differs from its README

The llama.cpp v1.1.0 server README describes some behaviour differently from what the server's source does. The library follows the source and the recorded behaviour of the real server. Evidence cites the llama.cpp source files at tag v1.1.0 or recordings from a real 1.1.0 server.

### Request parameters and response fields

| README says | Server does | Evidence |
|---|---|---|
| `samplers` is an array of names | Also accepts a compact string such as `"kpt"` | server-schema.cpp:472-479 |
| Speculative request parameters appear in examples | Their setters are compiled out with `#if 0` | server-schema.cpp:199-229 |
| `response_fields` `"a/b"` moves `b` to the top under a new name | The top-level member keeps the literal name `"a/b"` | server-task.cpp:354; recorded from the real server |
| `json_schema` and `grammar` are separate options | With both present the server uses `grammar` | server-schema.cpp:276-299; recorded from the real server |
| A prompt object is `{prompt_string: "string", multimodal_data: [...]}` | Without media, `prompt_string` may also be a token array or a mixed array | server-common.cpp:1222-1225, 1084-1115; recorded from the real server |
| `reasoning_effort`: any value other than `none` is passed to the template | Values the template does not allow are rejected with 400 | server-common.cpp:1654-1659 |
| `tokens_cached` is the number of prompt tokens reused from the cache | It is the size of the slot cache when generation ends, generated tokens included | server-context.cpp:2415-2416 |

### Completion, infill, tokenize and apply-template

| README says | Server does | Evidence |
|---|---|---|
| `generation_settings` of `/completion` includes `n_ctx` and `model` | Neither is in it | README 681; recorded from the real server |
| `/infill` `prompt` is added after the FIM_MID token | It is appended to the input prefix, before the suffix | server-common.cpp:2086-2103; recorded from the real server |
| Tokenize `content` is required text; the `"á"` example lists two tokens | `content` may be absent or a mixed list; the real output has a leading whitespace token before the two byte pieces | server-context.cpp:6419-6457; recorded from the real server |
| `/apply-template` takes `messages` | The handler runs the full chat parser (tools, tool choice, response format, continuation) and never streams | server-context.cpp:6391-6404; server-common.cpp:1498-1698 |
| Native stream partials carry only `content`, `tokens`, and `stop` | Partials also carry `index`, token counters, `id_slot`, and optional progress, timings and probabilities | README 626; server-task.cpp:1052-1073 |
| POST `/completion` documents `repeat_penalty` default 1.1 and a sampler chain without `penalties` and `top_n_sigma` | The default is 1.0 and the chain includes both, matching the CLI table and recorded `/props` responses | README 124-137; recorded from the real server |

### Embeddings and reranking

| README says | Server does | Evidence |
|---|---|---|
| The rerank score is a match probability in [0, 1] | The server copies the model's output without enforcing a range and uses -1e6 when embeddings are missing | README 849; server-context.cpp:2595-2607 |
| `texts` selects TEI, `documents` selects Jina (only the unmixed cases are described) | With both keys present, `texts` selects the TEI output shape while `documents` wins as the input | README 802-803; server-context.cpp:6892, 6957-6962 |
| `/embedding` takes `content`; `/embeddings` supports every pooling | They are the same handler: `/embedding` also accepts `input` (which wins over `content`) and every pooling | server.cpp:198-200; server-context.cpp:6761-6875 |
| Native embeddings take the same options as `/v1/embeddings` | The native endpoints validate `encoding_format: base64` and then ignore it | server-context.cpp:6761-6875; recorded from the real server |
| Pooled embeddings are normalised with the Euclidean norm | Both pooled endpoints honour `embd_normalize` | server-context.cpp:6761-6875 |
| `/v1/embeddings` returns the OpenAI envelope | Given the legacy `content` member it answers with the native envelope; `model` in the request overrides the echoed name; `dimensions` is not read | server-common.cpp:1722-1763; recorded from the real server |
| Embeddings support `encoding_format: base64` | The encoded bytes use the server's native float byte order, not a fixed byte order | server-common.cpp:1733-1738 |

### Props, slots and metrics

| README says | Server does | Evidence |
|---|---|---|
| `/slots/{id_slot}?action=save\|restore\|erase` are three endpoints | One route `POST /slots/:id_slot`, action in the query | server.cpp:220 |
| The `/props` example has no `voices` member | The handler writes `voices`; the text below the example documents it | README 953-1048; server-context.cpp:5691 |
| `default_generation_settings` of `/props` equals the full generation settings and shows slot state | It holds only `{params: <reduced subset>, n_ctx}`; in router mode `params` is null | README 961-1046; recorded from the real server |
| Examples show `speculative.n_max`, `speculative.n_min`, `speculative.p_min` | Settings serialise only `speculative.types` | README 1014, 1162, 1227; recorded from the real server |
| Slot examples show `next_token` as an object | The server emits a one-element array, and a fresh slot omits task-specific members | README 1169, 1234; recorded from the real server |
| The metrics table lists 15 metrics | The server also emits `prompt_tokens_cached_total`, and a `Process-Start-Time-Unix` response header | server-task.cpp:1581-1583; server-context.cpp:5742-5802 |

### LoRA adapters

| README says | Server does | Evidence |
|---|---|---|
| `--lora-init-without-apply` starts all adapters at scale 0 | The real server reports scale 1; `POST []` sets 0. The flag skips applying the adapter at start but does not change the stored scale the list reports | recorded from the real server; common.cpp:1508 |
| `GET /lora-adapters` returns `id`, `path`, `scale` | The server also writes `task_name` and `prompt_prefix`, and `alora_invocation_string` / `alora_invocation_tokens` when present | server-task.cpp:1711-1728 |
| `POST /lora-adapters` response is not described | It answers `{"success": true}`; a non-array body is a 400 | server-task.cpp:1735-1736; server-context.cpp:6626-6657 |
| LoRA adapter updates can be routed like other POST endpoints | In router mode `POST /lora-adapters?model=...` returns 400: the router reads `model` from an object body, but the endpoint needs an array | server-models.cpp:2165-2172; server-context.cpp:6626-6631; recorded from the real server |

### OpenAI completions, chat, Responses and Anthropic Messages

| README says | Server does | Evidence |
|---|---|---|
| `/v1/completions` follows the OpenAI Completions API | The handler skips the OpenAI parameter parser: without a projector it accepts native text and token prompt forms, `logprobs` is a count, and `echo`, `best_of` and `suffix` are ignored | server-context.cpp:5993-6003, server-common.cpp:1252-1289 |
| Usage in the last stream chunk depends on `stream_options` | `/v1/completions` always emits `usage` on the final chunk | server-task.cpp:366-404 |
| Tool definitions and `tool_choice` as in the OpenAI API | `tool_choice` also accepts an `allowed_tools` form; tools accept `defer_loading`; `strict` is ignored | common/chat.cpp:339-386, 705-739 |
| `/control` on a finished completion is a no-op | It answers 200 with `success: false` and a message | recorded from the real server |
| Anthropic `model` is required; named `tool_choice` selects a tool; `temperature` and `top_p` default to 1.0 | `model` is not read on a single-model server; `tool` and `any` both mean any required tool; omitted settings use the server defaults; `stop_sequences` may be a string | server-chat.cpp:598-687; server-common.cpp:1512-1517; recorded from the real server |
| Responses follow the OpenAI Responses API | Stored and previous responses, non-function tools and file input are rejected or skipped; the non-stream output has `completed_at`, the final stream snapshot does not | server-chat.cpp:6-352; recorded from the real server |
| `/v1/completions` accepts OpenAI completion prompts and supports native completion features | With a multimodal projector loaded, the route accepts only a string prompt. It does not accept multimodal object prompts with or without a projector | README 1430-1438; server-context.cpp:5332-5337, 5993-6003; server-common.cpp:1210-1228 |

### Speech and transcription

| README says | Server does | Evidence |
|---|---|---|
| `/v1/audio/transcriptions` is mentioned only in passing | The route exists | server.cpp:190-191 |
| `/tts` `response_format` is one of wav, pcm, pcm16 | An unknown value yields WAV instead of an error; `max_tokens` is an undocumented alias of `n_predict` | server-schema.cpp:537-580 |
| `/v1/audio/speech` PCM is mono; omitted `stream_format` means buffered | The model's channel count is kept; a native `stream: true` survives the conversion | server-chat.cpp:785-815; server-context.cpp:7145-7219 |
| `/tts` multipart upload takes the same options as JSON | The multipart parser converts `stream` and numbers but not `raw_prompt`, which silently becomes false | server-http.cpp:825-839; server-context.cpp:5706-5722 |

### Realtime

| README says | Server does | Evidence |
|---|---|---|
| Realtime `max_output_tokens` is an integer or `"inf"` | Only positive integers (capped at INT32_MAX) or `"inf"` are accepted | server-realtime.cpp:37-123 |
| Realtime client events are the ones listed per section | Both handlers also accept `transcription_session.update`; the conversation handler accepts `input_audio_buffer.clear` and answers `input_audio_buffer.cleared`; `session.update` also takes `audio.output.voice` (name or `{id}`) and `audio.output.language`; `output_modalities: null` resets to automatic | server-realtime.cpp:198-245, 389-430, 677-777; recorded from the real server |
| Realtime `error` events carry `error.message` | The specific reason (for example `conversation_empty`) is in `error.type`; `code` and `param` are null | recorded from the real server |

### System One

| README says | Server does | Evidence |
|---|---|---|
| System One `state` may be null for lfm2-d1 and lfm2-d1-omni | The source also accepts null for vllm-d3 and both lfm3-d1 variants | README 1929; server-decision.cpp:236-253 |
| A `score` answer is the expected level index | The lfm3-d1 variants return the median | README 1974; server-decision.cpp:1760-1812 |
| `noul` criteria describe `true` and `false` | The source also accepts model-specific no/yes aliases. It reads both `files` and `images`, so sending both doubles the input | server-decision.cpp:254-382, 389-477 |
| System One choice `confidence` of zero means all options are equally likely | The lfm3-d1 variants report the maximum probability instead (0.5 for two equally likely options) | server-decision.cpp:1701-1711 |
| System One question `instructions` may be omitted for h2o-lightning and OCR models | The source also accepts omitted instructions for vllm-d3 | server-decision.cpp:265-273 |

### Model lists and the router

| README says | Server does | Evidence |
|---|---|---|
| `status` of a model in `GET /models`: `unloaded` has no `args`; `downloading` carries nested `progress` | `status` always has `args` (and often `preset`), also when unloaded; the list does not emit download progress | server-models.cpp:2214-2341; recorded from the real server |
| `/models/sse` download progress is `data[URL]`; status updates are `model_status` events | Progress is `data.progress[URL]`; updates use `status_change` except the first router `model_status` event | server-models.cpp:1458-1522, 1965-2012; recorded from the real server |
| `GET /v1/models` returns `object` and `data` | A single-model server also returns a separate Ollama-style `models` array next to `data` | server-context.cpp:5586-5644; recorded from the real server |
| `DELETE /models` answers `{"success": true}` | Deleting a sidecar that is not there can answer 200 with `success: false` | recorded from the real server |
| Router `memory.effective` is null unless the model is loaded | The value of the last load is kept while the model sleeps and is cleared only on unload | README 2251; server-models-sched.cpp:354-365, 553-555; recorded from the real server |

## License

Apache License 2.0. See [LICENSE](LICENSE).
