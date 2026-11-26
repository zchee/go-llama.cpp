module github.com/zchee/go-llama.cpp/integration

go 1.27

replace github.com/zchee/go-llama.cpp => ../

require (
	github.com/zchee/go-llama.cpp v0.0.0-00010101000000-000000000000
	go.uber.org/goleak v1.3.0
	go.yaml.in/yaml/v4 v4.0.0
)

require (
	github.com/cenkalti/backoff/v5 v5.0.3 // indirect
	github.com/coder/websocket v1.8.15 // indirect
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/tmaxmax/go-sse v0.12.0 // indirect
)
