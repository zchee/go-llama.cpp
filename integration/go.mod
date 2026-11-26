module github.com/zchee/go-llama.cpp/integration

go 1.27

replace github.com/zchee/go-llama.cpp => ../

require (
	github.com/coder/websocket v1.8.15
	github.com/google/go-cmp v0.7.0
	github.com/zchee/go-llama.cpp v0.0.0-00010101000000-000000000000
	go.uber.org/goleak v1.3.0
	go.yaml.in/yaml/v4 v4.0.0
)

require (
	github.com/cenkalti/backoff/v5 v5.0.3 // indirect
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/pandatix/go-abnf v0.5.5 // indirect
	github.com/prometheus/client_model v0.6.3 // indirect
	github.com/prometheus/common v0.74.0 // indirect
	github.com/tmaxmax/go-sse v0.12.0 // indirect
	google.golang.org/protobuf v1.36.12 // indirect
)
