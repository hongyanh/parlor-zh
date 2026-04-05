# Open Source References — Real-Time Multimodal AI Demos

## Most Relevant Projects

### Full Pipeline (Camera + Mic + LLM + TTS)

#### 1. Vocalis — Speech-to-Speech AI with Vision
- **GitHub**: https://github.com/Lex-au/Vocalis
- **Stars**: ~294
- **Stack**: React/TypeScript frontend, FastAPI/Python backend, WebSocket
- **Models**: LLaMA 3 8B (LM Studio), SmolVLM-256M (vision), Faster-Whisper (STT), Orpheus TTS
- **Features**: Mid-speech interruption, AI-initiated follow-ups, image analysis, low-latency audio streaming
- **Relevance**: One of few projects combining ALL four elements. Good architectural reference for React + FastAPI + WebSocket.

#### 2. MiniCPM-o — End-to-End Multimodal with Full-Duplex Streaming
- **GitHub**: https://github.com/OpenBMB/MiniCPM-o
- **Stars**: ~24,300
- **Stack**: PyTorch, llama.cpp, Ollama, vLLM, SGLang, WebRTC
- **Models**: MiniCPM-o 4.5 (9B), SigLip2 vision, Whisper-medium audio, CosyVoice2 TTS
- **Features**: Full-duplex live streaming, simultaneous video/audio input with speech output, proactive scene interaction
- **Relevance**: Best architectural reference for end-to-end multimodal streaming.

#### 3. Open-LLM-VTuber — Voice + Vision AI Companion
- **GitHub**: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
- **Stars**: ~6,400
- **Stack**: Python, JavaScript, Live2D, Docker
- **Models**: Ollama/OpenAI/Gemini (LLM), Whisper/FunASR (STT), 10+ TTS engines
- **Features**: Fully offline, camera/screen perception, Live2D avatar
- **Relevance**: Shows how to combine camera perception with voice using local models via Ollama.

### Gemma-Specific Voice Demos (Audio only, no camera)

#### 4. macos-local-voice-agents — THE closest reference
- **GitHub**: https://github.com/kwindla/macos-local-voice-agents
- **Stack**: **Pipecat** + Gemma 3n E4B + MLX Whisper + Kokoro TTS on **macOS**
- **Features**: Silero VAD, Smart Turn v2, SmallWebRTCTransport, subprocess-isolated TTS
- **Relevance**: **Directly applicable**. Same platform (macOS), same model family (Gemma), same framework (Pipecat). Just needs camera added and upgrade to Gemma 4.

#### 5. Local-Voice-AI-Agent — FastRTC + Gemma
- **GitHub**: https://github.com/Utkarsh4412/Local-Voice-AI-Agent
- **Stars**: ~1
- **Stack**: FastRTC + Gemma 3 (Ollama) + Moonshine STT + Kokoro TTS
- **Relevance**: Smallest codebase, proves FastRTC + Gemma works.

#### 6. mOrpheus — Gemma + Orpheus TTS
- **GitHub**: https://github.com/Nighthawk42/mOrpheus
- **Stars**: ~85
- **Stack**: Gemma 3 12B (LM Studio) + Whisper STT + Orpheus 3B TTS
- **Relevance**: Clean Gemma + TTS integration reference.

### Framework-Level References

#### 7. Pipecat — Production Multimodal AI Framework
- **GitHub**: https://github.com/pipecat-ai/pipecat
- **Stars**: ~11,000
- **Features**: 10+ LLM providers, 10+ STT engines, 20+ TTS engines, WebRTC transport
- **Built-in services**: `OLLamaLLMService`, `KokoroTTSService`, `PiperTTSService`, `WhisperSTTServiceMLX`
- **Key examples**:
  - `examples/function-calling/function-calling-moondream-video.py` — camera + audio + vision LLM + TTS
  - `examples/function-calling/function-calling-google-video.py` — same with Gemini
  - `examples/function-calling/function-calling-ollama.py` — Ollama integration

#### 8. FastRTC — Gradio Real-Time Communication
- **GitHub**: https://github.com/gradio-app/fastrtc
- **Stars**: ~4,600
- **Key example**: `demo/gemini_audio_video/app.py` — the canonical pattern for webcam + mic + LLM
- **Pattern**: Subclass `AsyncAudioVideoStreamHandler`, implement `receive()`, `emit()`, `video_receive()`, `video_emit()`

#### 9. LiveKit Agents
- **GitHub**: https://github.com/livekit/agents
- **Stars**: ~9,900
- **Features**: Production WebRTC voice AI agents, STT/LLM/TTS pipelines

### Vision-Only References

#### 10. SmolVLM Real-time Webcam
- **GitHub**: https://github.com/ngxson/smolvlm-realtime-webcam
- **Stars**: ~5,500
- **What**: Minimal webcam → VLM loop (single HTML file + llama.cpp server)
- **Relevance**: Simplest reference for continuous webcam-to-VLM inference.

#### 11. NVIDIA Live VLM WebUI
- **GitHub**: https://github.com/NVIDIA-AI-IOT/live-vlm-webui
- **Stars**: ~300
- **What**: Universal web UI for real-time VLM interaction via webcam

### Voice-Only References

#### 12. RealtimeVoiceChat
- **GitHub**: https://github.com/KoljaB/RealtimeVoiceChat
- **Stars**: ~3,600
- **Stack**: FastAPI + WebSocket + Ollama + Kokoro/Coqui TTS
- **Relevance**: Clean voice interaction loop reference.

#### 13. HuggingFace Speech-to-Speech
- **GitHub**: https://github.com/huggingface/speech-to-speech
- **Stars**: ~4,600
- **Stack**: Silero VAD → Whisper → any HF model → MeloTTS/Kokoro
- **Features**: MLX support for Apple Silicon

## FastRTC Gemini Audio Video — Full Source Code

This is the exact pattern we'd adapt for our demo:

```python
from fastrtc import AsyncAudioVideoStreamHandler, Stream, wait_for_item

class GeminiHandler(AsyncAudioVideoStreamHandler):
    def __init__(self):
        super().__init__("mono", output_sample_rate=24000, input_sample_rate=16000)
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()

    async def start_up(self):
        # Initialize LLM connection, start receive loop
        pass

    async def receive(self, frame: tuple[int, np.ndarray]):
        # Handle incoming audio from mic
        _, array = frame
        await self.session.send(input=encode_audio(array.squeeze()))

    async def emit(self):
        # Return audio to play back (TTS output)
        array = await wait_for_item(self.audio_queue, 0.01)
        return (self.output_sample_rate, array) if array is not None else None

    async def video_receive(self, frame: np.ndarray):
        # Handle incoming video frames (rate-limit to 1fps)
        if time.time() - self.last_frame_time > 1:
            self.last_frame_time = time.time()
            await self.session.send(input=encode_image(frame))

    async def video_emit(self):
        # Mirror webcam frames back to UI
        frame = await wait_for_item(self.video_queue, 0.01)
        return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

stream = Stream(handler=GeminiHandler(), modality="audio-video", mode="send-receive")
stream.ui.launch()
```

## Pipecat Camera + Audio Pipeline Pattern

From `function-calling-moondream-video.py`:

```python
transport = SmallWebRTCTransport(
    params=TransportParams(
        video_in_enabled=True,   # Enable webcam
        audio_in_enabled=True,   # Enable mic
        audio_out_enabled=True,  # Enable speaker
    )
)

# Pipeline: transport → STT → LLM → TTS → transport
pipeline = Pipeline([
    transport.input(),
    stt,                          # Whisper/Deepgram
    context_aggregator.user(),
    llm,                          # Gemma 4 via Ollama
    tts,                          # Kokoro
    transport.output(),
    context_aggregator.assistant(),
])

# Camera frame capture on demand
await maybe_capture_participant_camera(transport, client)
```
