# Gemma 4 Model Overview

## Model Family

Gemma 4 was released April 2, 2026 by Google DeepMind under **Apache 2.0** license.

| Variant | Total Params | Effective Params | Context | Modalities | VRAM (4-bit) | VRAM (bf16) |
|---------|-------------|-----------------|---------|------------|-------------|-------------|
| **E2B** | 5.1B | 2.3B | 128K | Text, Image, Video, **Audio** | 4-5 GB | 10-15 GB |
| **E4B** | 8B | 4.5B | 128K | Text, Image, Video, **Audio** | 5.5-6 GB | ~16 GB |
| **26B-A4B (MoE)** | 26B | 3.8B active | 256K | Text, Image, Video | 16-18 GB | ~52 GB |
| **31B (Dense)** | 31B | 31B | 256K | Text, Image, Video | 17-20 GB | ~62 GB |

**Key**: Only E2B and E4B support native audio input. The "E" stands for "effective" — Per-Layer Embeddings (PLE) make the compute cost much lower than the total parameter count.

## E4B Architecture

### Text Decoder
- **Layers**: 42
- **Hidden size**: 2560
- **Intermediate (FFN) size**: 10,240
- **Per-layer input hidden size**: 256 (PLE feature)
- **Attention heads**: 8, KV heads: 2 (grouped-query attention)
- **Head dimension**: 256 (sliding), 512 (global)
- **Vocabulary size**: 262,144
- **Context length**: 128K tokens
- **Hybrid attention**: 5 sliding window (512 tokens) + 1 global, repeating 7 times
- **KV sharing**: across 18 layers

### Vision Encoder (~150M params)
- **Layers**: 16
- **Hidden size**: 768
- **Patch size**: 16
- **Default output**: 280 tokens per image
- **Configurable token budgets**: 70, 140, 280, 560, 1120
  - 70: classification, fast video
  - 140: quick understanding
  - 280: general multimodal chat
  - 560: complex UI reasoning
  - 1120: OCR, documents, handwriting

### Audio Encoder (~300M params)
- **Layers**: 12
- **Hidden size**: 1,024
- **Feature extractor**: 128 Mel-frequency bins, 16kHz sampling rate
- **Processing**: 40ms per token, max 750 tokens = **30 seconds max**
- **Activation**: SiLU

### Video Processing
- Frames sampled: 32
- Max 70 soft tokens per frame
- Max duration: 60 seconds at 1 fps

## Special Tokens

| Token | String | Purpose |
|-------|--------|---------|
| Image | `<\|image\|>` | Image data placeholder |
| Audio | `<\|audio\|>` | Audio data placeholder |
| Video | `<\|video\|>` | Video data placeholder |
| Think | `<\|think\|>` | Enable reasoning mode |
| BOI/EOI | `<\|image>` / `<image\|>` | Image region markers |
| BOA/EOA | `<\|audio>` / `<audio\|>` | Audio region markers |
| Tool call | `<\|tool_call>` / `<tool_call\|>` | Function calling |

## Thinking/Reasoning Mode

- Enable by adding `<|think|>` at start of system prompt
- Output format: `<|channel>thought\n[reasoning]<channel|>[answer]`
- **Do NOT use in real-time demo** — too slow for conversational latency

## Inference Code (Transformers)

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E4B-it", dtype="auto", device_map="auto")

# Multimodal message format
messages = [
    {"role": "user", "content": [
        {"type": "image", "url": "https://example.com/photo.jpg"},
        {"type": "audio", "audio": "https://example.com/audio.wav"},
        {"type": "text", "text": "Describe what you see and hear."}
    ]}
]

inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, top_p=0.95, top_k=64)
```

## Recommended Sampling Parameters
- Temperature: 1.0
- Top-p: 0.95
- Top-k: 64

## Best Practices
1. Place image/audio content BEFORE text in prompts
2. Audio max: 30 seconds per chunk
3. Video max: 60 seconds at 1 fps
4. Use lower vision token budgets (70-140) for speed, higher (560+) for detail
5. Do NOT enable thinking mode for real-time applications

## Benchmarks (E4B instruction-tuned)

| Benchmark | Score |
|-----------|-------|
| MMLU Pro | 69.4% |
| AIME 2026 | 42.5% |
| LiveCodeBench v6 | 52.0% |
| GPQA Diamond | 58.6% |
| MMMU Pro (vision) | 52.6% |
| MATH-Vision | 59.5% |
| MMMLU (multilingual) | 76.6% |

## Sources
- https://huggingface.co/google/gemma-4-E4B
- https://huggingface.co/google/gemma-4-E4B-it
