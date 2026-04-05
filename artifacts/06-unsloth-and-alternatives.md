# Unsloth & Alternative Inference Options

## Unsloth — Quantized Gemma 4 Models

Unsloth provides optimized quantized versions of all Gemma 4 variants.

### Available Formats

| Model | GGUF (llama.cpp) | MLX (Apple Silicon) |
|-------|-------------------|---------------------|
| E2B | `unsloth/gemma-4-E2B-it-GGUF` (Q8_0) | Available |
| E4B | `unsloth/gemma-4-E4B-it-GGUF` (Q8_0) | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` |
| 26B-A4B | `unsloth/gemma-4-26B-A4B-it-GGUF` (UD-Q4_K_XL) | Available |
| 31B | `unsloth/gemma-4-31B-it-GGUF` (UD-Q4_K_XL) | Available |

"UD" = Unsloth Dynamic quantization (selectively quantizes layers at different bit levels).

### GGUF Inference (llama.cpp)

```bash
# Build llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON
cmake --build llama.cpp/build --config Release -j

# Run E4B
./llama.cpp/build/bin/llama-cli \
    -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
    --temp 1.0 --top-p 0.95 --top-k 64

# With vision (needs mmproj file)
hf download unsloth/gemma-4-E4B-it-GGUF \
    --include "*mmproj-BF16*" --include "*Q8_0*"
./llama.cpp/build/bin/llama-cli \
    --model gemma-4-E4B-it-Q8_0.gguf \
    --mmproj mmproj-BF16.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64
```

### MLX Inference (Apple Silicon)

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/install_gemma4_mlx.sh | sh
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate

# Run
python -m mlx_lm chat --model unsloth/gemma-4-E4B-it-UD-MLX-4bit --max-tokens 4096
```

### Ollama

```bash
ollama pull gemma4:e4b
ollama run gemma4:e4b
```

Provides OpenAI-compatible API at `http://localhost:11434/v1`.

## Comparison Table for M3 Pro 18GB

| Backend | Model Size | Vision | Audio | GPU | Setup | Ecosystem |
|---------|-----------|--------|-------|-----|-------|-----------|
| **LiteRT-LM** | **3.65 GB** | **Yes** | **Yes** | **Metal (C++)** | Bazel build | Small (10 models) |
| Ollama | ~6 GB | Yes | No | Metal | One command | Huge |
| llama.cpp | ~6 GB (Q8) | Yes (mmproj) | No | Metal | cmake | Huge (GGUF) |
| MLX | ~5 GB (4-bit) | Yes (mlx-vlm) | No | Metal | pip install | Growing |
| Transformers | ~16 GB (bf16) | Yes | Yes | No Metal | pip install | Everything |

## Why We Chose LiteRT-LM Over Alternatives

1. **Smallest model**: 3.65 GB vs 5-6 GB — more headroom on 18 GB machine
2. **Native audio**: No separate Whisper STT needed — simpler pipeline
3. **Google-optimized**: Their engine, their model, their quantization
4. **Metal GPU**: C++ API has full Metal acceleration
5. **Production quality**: Powers Chrome and Pixel Watch

## Fallback Plan

If LiteRT-LM has issues (build problems, missing E4B model, poor M3 Pro performance):

**Fallback 1: Ollama** — `ollama pull gemma4:e4b`, use with Pipecat's `OLLamaLLMService`. Add faster-whisper for STT since Ollama doesn't support audio input.

**Fallback 2: MLX** — Best Apple Silicon performance. Use `mlx-vlm` for vision, `mlx-whisper` for STT. More Python code needed.

**Fallback 3: llama.cpp server** — `llama-server -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 --mmproj mmproj-BF16.gguf`. OpenAI-compatible API. Add Whisper for STT.
