# Architecture & Implementation Plan

## Target Hardware
- **Apple M3 Pro, 18 GB unified memory**
- macOS

## Chosen Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **LLM Inference** | LiteRT-LM C++ (Metal GPU) | Fastest, smallest model (3.65 GB), native audio+vision |
| **Model** | Gemma 4 E4B-it (.litertlm) | Native multimodal: text + image + audio |
| **TTS** | Kokoro-82M | <300ms, 82M params, excellent quality |
| **Web Frontend** | HTML/JS + WebRTC or WebSocket | Camera + mic capture via getUserMedia |
| **HTTP Server** | C++ with cpp-httplib (or similar) | Wraps LiteRT-LM engine, serves SSE/WebSocket |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    Browser                           │
│                                                      │
│  getUserMedia() → Webcam (1fps JPEG) + Mic (PCM)   │
│       │                                    │         │
│       └──── WebSocket ─────────────────────┘         │
│                    │                                  │
└────────────────────┼──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              C++ Server (Metal GPU)                   │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │        LiteRT-LM Engine                      │    │
│  │        Gemma 4 E4B (3.65 GB, Metal GPU)     │    │
│  │                                              │    │
│  │  Input:                                      │    │
│  │   • Audio blob (WAV) — native, no STT needed │    │
│  │   • Image blob (JPEG) — webcam frame         │    │
│  │   • Text — conversation context              │    │
│  │                                              │    │
│  │  Output:                                     │    │
│  │   • Streaming text tokens                    │    │
│  └──────────────┬──────────────────────────────┘    │
│                  │ streamed text                      │
│                  ▼                                    │
│  ┌─────────────────────────────────────────────┐    │
│  │        Kokoro TTS (82M, ONNX)               │    │
│  │        Sentence-level streaming              │    │
│  │        Output: PCM audio chunks              │    │
│  └──────────────┬──────────────────────────────┘    │
│                  │                                    │
└──────────────────┼────────────────────────────────────┘
                   │ audio (WebSocket binary frames)
                   ▼
              Browser Speaker (Web Audio API)
```

## Memory Budget (M3 Pro 18 GB)

| Component | Estimated Memory |
|-----------|-----------------|
| Gemma 4 E4B (.litertlm, Metal) | ~3.65 GB model + ~1.5 GB GPU working | 
| Kokoro TTS (ONNX) | ~200 MB |
| C++ server + buffers | ~200 MB |
| Browser + OS | ~4 GB |
| **Total** | **~9.5 GB** |
| **Headroom** | **~8.5 GB** |

## Data Flow

### User speaks + shows camera → AI responds with voice

1. **Browser** captures webcam frame (JPEG, ~50-100 KB) at 1 fps
2. **Browser** captures microphone audio, detects silence (VAD in JS or server-side)
3. On user turn end: sends audio chunk (WAV, ~30s max) + latest frame via WebSocket
4. **C++ Server** receives both, constructs multimodal message:
   ```json
   {
     "role": "user",
     "content": [
       {"type": "audio", "blob": "<base64_wav>"},
       {"type": "image", "blob": "<base64_jpeg>"},
       {"type": "text", "text": "The user is speaking to you while showing their camera. Respond conversationally."}
     ]
   }
   ```
5. **LiteRT-LM** processes with Gemma 4 E4B (Metal GPU), streams tokens
6. **Kokoro TTS** converts first complete sentence to audio while LLM continues
7. **Server** sends audio chunks back via WebSocket
8. **Browser** plays audio via Web Audio API

## Measured Benchmarks (M3 Pro 18GB, Metal GPU)

Tested with `litert_lm_main`, Gemma 4 E4B (3.4 GB), Metal native backend:

| Metric | Value |
|--------|-------|
| **TTFT** | **0.38s** |
| **Prefill** | **61.3 tok/s** |
| **Decode** | **26.5 tok/s** |
| Init time | ~5.9s (one-time) |
| 140 token output | 5.3s total |

**IMPORTANT**: Must symlink `libLiteRtMetalAccelerator.dylib` → `libLiteRtGpuAccelerator.dylib` for native Metal. Without it, falls back to WebGPU (19 tok/s, 1.18s TTFT).

## Latency Targets (Revised with Real Data)

| Stage | Target | Notes |
|-------|--------|-------|
| Audio capture + VAD | ~200ms after speech ends | Silero VAD or browser-side |
| WebSocket round-trip | <10ms | localhost |
| LiteRT-LM TTFT (GPU) | **~380ms** | Measured on M3 Pro Metal |
| LiteRT-LM decode | **~26.5 tok/s** | Measured on M3 Pro Metal |
| First sentence complete | ~1s | ~26 tokens at 26.5 tok/s |
| Kokoro TTS first audio | <300ms | From first complete sentence |
| **Total perceived latency** | **~1.5-2s** | From user stops speaking to AI voice starts |

## Implementation Phases

### Phase 1: Verify LiteRT-LM Build & GPU Inference
1. Clone LiteRT-LM to ~/workspace
2. Build with Bazel on macOS
3. Download Gemma 4 E4B model
4. Verify text generation with GPU backend
5. Test multimodal (image + text) inference
6. Test audio input

### Phase 2: C++ HTTP/WebSocket Server
1. Add cpp-httplib or similar lightweight HTTP library
2. Implement WebSocket endpoint for streaming
3. Accept multimodal input (audio blob + image blob)
4. Stream LLM tokens back as SSE or WebSocket messages
5. Integrate Kokoro TTS (via ONNX C++ runtime or subprocess)

### Phase 3: Web Frontend
1. HTML/JS page with getUserMedia for webcam + mic
2. WebSocket connection to C++ server
3. Client-side VAD (Silero WASM or simple energy-based)
4. Frame sampling at 1 fps
5. Web Audio API for playing TTS audio
6. Simple chat UI showing conversation

### Phase 4: Polish & Optimize
1. Tune vision token budget (start with 70 for speed)
2. Optimize audio chunk size for latency
3. Add conversation history management
4. Handle interruption (cancel generation when user starts speaking)
5. Add system prompt tuning for natural conversation

## Open Questions / Risks

1. **Bazel build on macOS** — may have dependency issues, need to verify
2. **LiteRT-LM Gemma 4 E4B model availability** — confirm .litertlm file exists for E4B (not just E2B)
3. **Audio quality via native input** — Gemma 4's audio encoder quality vs dedicated Whisper STT
4. **Kokoro TTS C++ integration** — may need Python subprocess or ONNX C++ runtime
5. **M3 Pro GPU performance** — no published benchmarks, need to measure
6. **WebSocket library in C++** — cpp-httplib doesn't support WebSocket natively; may need uWebSockets, Boost.Beast, or similar

## Alternative: Hybrid Python + C++ Approach

If pure C++ is too complex for the server layer:

```
Browser ←→ Python FastAPI (WebSocket, Kokoro TTS)
                ↕
           C++ LiteRT-LM Engine (Metal GPU)
           via: subprocess CLI | shared memory | Unix socket
```

This lets us use Python for the "glue" (WebSocket, TTS, audio encoding) while keeping the hot path (LLM inference) in C++ with Metal GPU. The `litert_lm_main` CLI already supports streaming output.

## Files to Create

```
gemma-4/
├── 01-gemma4-model-overview.md      # Model documentation
├── 02-litert-lm-guide.md            # LiteRT-LM reference
├── 03-open-source-references.md     # Related projects
├── 04-tts-options.md                # TTS comparison
├── 05-architecture-and-plan.md      # This file
├── 06-unsloth-and-alternatives.md   # Unsloth/Ollama/MLX info
└── src/                             # Demo source code (TBD)
    ├── server/                      # C++ server wrapping LiteRT-LM
    ├── frontend/                    # HTML/JS web UI
    └── tts/                         # Kokoro TTS integration
```
