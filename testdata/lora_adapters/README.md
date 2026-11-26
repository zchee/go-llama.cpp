# LoRA Adapter Activation Fixture Recipe

The `activation*` fixtures (`activation.resp.json`, `activation_set.req.json`,
`activation_set.resp.json`, and `activation_scaled.resp.json`) require an adapter carrying aLoRA
activation and task metadata, which none of the upstream test adapters provide directly.

## Pinned Inputs and Tooling

- Base model (`stories15M-MOE` in `integration/models.yaml`):
  - Repository: `ggml-org-dev/stories15M_MOE`
  - Revision: `b79d85fd44cb281d525c0f75e79b522b9fdf714b`
  - File: `stories15M_MOE-F16.gguf`
  - SHA-256: `1240dfc1957df9f3550dd6c1d9e64b466fc2f452d8bc34bd4e45e1a1e2ca6055`
- Base LoRA adapter (`stories15M-MOE-lora` in `integration/models.yaml`):
  - Repository: `ggml-org-dev/stories15M_MOE`
  - Revision: `b79d85fd44cb281d525c0f75e79b522b9fdf714b`
  - File: `moe_shakespeare15M.gguf`
  - SHA-256: `d1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a`
- GGUF tool: `uv run --script testdata/lora_adapters/make_activation_adapter.py` using Python package `gguf==0.19.0`.
- Added GGUF metadata keys on the modified copy (`invocation.gguf`):
  - `adapter.alora.invocation_tokens`: `ARRAY` of `UINT32` = `[42, 43]` (decodes to `alora_invocation_string` = `"'("` under `stories15M_MOE`)
  - `adapter.lora.task_name`: `STRING` = `"fixture task"`
  - `adapter.lora.prompt_prefix`: `STRING` = `"fixture prefix"`

## Regeneration Steps

Run from the repository root. Use a temporary working directory with relative filenames so
`llama-server` records `"moe_shakespeare15M.gguf"` and `"invocation.gguf"` instead of local paths:

```sh
repo="$(pwd)"
workdir="$(mktemp -d)"
cache="${LLAMA_MODEL_CACHE:-$HOME/Library/Caches/go-llama.cpp/models}/ggml-org-dev/stories15M_MOE/b79d85fd44cb281d525c0f75e79b522b9fdf714b"

ln -s "$cache/stories15M_MOE-F16.gguf" "$workdir/stories15M_MOE-F16.gguf"
ln -s "$cache/moe_shakespeare15M.gguf" "$workdir/moe_shakespeare15M.gguf"
uv run --script "$repo/testdata/lora_adapters/make_activation_adapter.py" \
  "$workdir/moe_shakespeare15M.gguf" "$workdir/invocation.gguf"

(
  cd "$workdir"
  exec llama-server --no-webui --log-colors off --offline \
    --model stories15M_MOE-F16.gguf --ctx-size 512 \
    --lora moe_shakespeare15M.gguf,invocation.gguf \
    --host 127.0.0.1 --port 18080
) &
srv_pid=$!
trap 'kill "$srv_pid" 2>/dev/null || true; wait "$srv_pid" 2>/dev/null || true; rm -rf "$workdir"' EXIT

until curl -sf http://127.0.0.1:18080/health >/dev/null; do sleep 0.2; done

go run ./integration/cmd/record -server http://127.0.0.1:18080 -method GET \
  -path /lora-adapters -out testdata/lora_adapters/activation
go run ./integration/cmd/record -server http://127.0.0.1:18080 -method POST \
  -path /lora-adapters -body '[{"id":1,"scale":0.75}]' -out testdata/lora_adapters/activation_set
go run ./integration/cmd/record -server http://127.0.0.1:18080 -method GET \
  -path /lora-adapters -out testdata/lora_adapters/activation_scaled
```
