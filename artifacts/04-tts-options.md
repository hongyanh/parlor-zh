# TTS (Text-to-Speech) Options

## Recommended: Kokoro-82M

- **Params**: 82M (tiny)
- **Speed**: <300ms for any text length, 96x real-time on GPU
- **Quality**: ELO 1,059 (highest-ranked open-weight model)
- **Languages**: EN, ES, FR, HI, IT, JA, PT, ZH
- **Runtime**: ONNX
- **Memory**: ~200 MB
- **Pipecat integration**: Built-in `KokoroTTSService`

```python
from pipecat.services.kokoro import KokoroTTSService
tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"))
```

## Alternatives

### Piper (~20M params)
- Fastest TTFB (~30ms). Runs on Raspberry Pi.
- Lower quality than Kokoro.
- Pipecat: `PiperTTSService`

### Orpheus 3B
- Higher quality, emotional speech with non-verbal cues.
- Much larger (3B params), slower.
- Used by mOrpheus project with Gemma.

### Dia-1.6B
- Non-verbal cues (laughter, breathing). Streaming.
- Larger than Kokoro, higher quality.

### edge-tts (Free, cloud)
- Uses Microsoft Edge's TTS API. No GPU needed.
- Good quality but requires network.
- `pip install edge-tts`

### ElevenLabs (Commercial API)
- Best quality overall.
- Network latency + cost.

## For Our Demo

**Kokoro-82M** is the clear choice:
- Runs locally on M3 Pro with minimal memory
- Sub-300ms latency
- Pipecat integration ready
- Quality is excellent for a demo

## Streaming TTS Architecture

The key optimization for perceived latency: **start TTS as soon as the first complete sentence arrives from the LLM**, not after the full response.

```
LLM output: "The sky is blue. | It reflects..." 
                              ↑
                    TTS starts here on first sentence
                    while LLM continues generating
```

Libraries that support this:
- **RealtimeTTS** (`pip install realtimetts[all]`) — wraps Kokoro, Piper, Edge, and more with automatic sentence chunking
- **Pipecat** — handles this natively in its pipeline architecture
