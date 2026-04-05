# UI Design Research: Gemma 4 Live Frontend

## What We're Building
A real-time voice + vision AI assistant demo running on-device (Gemma 4 on Apple Silicon). The user speaks into a microphone, shows things to a webcam, and the AI responds with voice + text. Key features: VAD-driven hands-free, barge-in support, tool calling, sub-3s latency.

## Competitive Landscape Analysis

### ChatGPT Advanced Voice Mode
- **Nov 2025 pivot**: Moved from separate orb screen → integrated voice in main chat thread
- Real-time transcription streams inline as AI speaks
- Tool results (maps, images, links) appear inline alongside voice
- Key insight: **Voice without text feels ungrounded** — users want to read what was said

### Google Gemini Live
- Material Design with fluid, ethereal four-color shapes
- Lottie animations synced to WebSocket audio events
- Tight animation↔audio synchronization is what makes it feel polished
- Generative UI: dynamic visual cards customized per query

### Apple Siri (iOS 18+)
- **Edge glow** around entire screen border — non-intrusive, content stays visible
- Glow responds to voice cadence (visual proof of listening)
- Haptic feedback on activation
- Key insight: **Ambient presence > screen takeover**

### Hume AI (Empathic Voice Interface)
- Real-time emotion analysis bar alongside conversation
- Prosody analysis (tone, rhythm, timbre) for natural turn-taking
- 300ms TTFB — feels instant
- Key insight: **Emotional visualization creates connection**

### ElevenLabs UI (Open Source)
- Three.js audio-reactive orb with four states (idle/listening/thinking/talking)
- 17 components: orb, waveform, bar visualizer, conversation bar, transcript viewer
- Word-by-word transcript highlighting synced to audio
- Generative UI: dynamic forms during voice conversations

### LiveKit Agents UI
- Five visualizer styles: Aura (WebGL), Wave, Radial, Grid, Bar
- Each has distinct behavior per state (connecting/speaking/listening/thinking)
- AgentChatTranscript + visualizer + controls in composable layout

### Perplexity Voice
- Voice as input method for search — results + citations appear while AI speaks
- "Voice lock" for ambient listening mode
- Sub-1-second first spoken response
- Key insight: **Showing sources alongside voice builds trust**

## Synthesized Design Principles

1. **Audio-reactive animation is the #1 feedback signal** — visuals must respond to actual audio energy, not just boolean states
2. **Four universal states**: Idle, Listening, Thinking, Speaking — every product distinguishes these
3. **Dark backgrounds** are near-universal for voice interfaces — makes glowing elements pop
4. **Real-time transcripts are now expected** — voice-only feels ungrounded (ChatGPT proved this)
5. **The orb/sphere is the dominant metaphor** — organic, alive, non-threatening
6. **Barge-in must feel instant** — abrupt visual transition, not smooth animation

## Design Alternatives

### A. "The Living Canvas" — Orb-Centric
```
┌─────────────────────────────────────┐
│                                     │
│           ┌───┐                     │
│           │cam│  (small PiP)        │
│           └───┘                     │
│                                     │
│         ╭──────────╮                │
│        ╱  LARGE ORB ╲              │
│       │  audio-reactive│            │
│        ╲   gradient   ╱             │
│         ╰──────────╯                │
│                                     │
│    "I see you at a desk..."         │
│                        LLM: 2.1s   │
│                                     │
└─────────────────────────────────────┘
```
- Full-screen dark bg, large centered audio-reactive orb
- Webcam as small circle PiP in corner
- Transcript fades in/out at bottom
- **Pros**: Visually stunning, great for demos
- **Cons**: De-emphasizes webcam (key differentiator), limited transcript

### B. "The Stage" — Camera-Forward with Overlay
```
┌─────────────────────────────────────┐
│ ╔═══════════════════════════════╗   │
│ ║                               ║   │
│ ║      FULL-SCREEN WEBCAM       ║   │
│ ║       (with edge glow)        ║   │
│ ║                               ║   │
│ ╠═══════════════════════════════╣   │
│ ║ ≋≋≋ waveform ≋≋≋  │ Listening║   │
│ ║ You: What do you see?        ║   │
│ ║ AI: I see a beautiful...     ║   │
│ ╚═══════════════════════════════╝   │
└─────────────────────────────────────┘
```
- Webcam fills the screen, edge glow indicates state (Siri-inspired)
- Semi-transparent overlay at bottom for transcript
- **Pros**: Camera front-and-center, immersive
- **Cons**: Text readability over video, less structured

### C. "The Companion" — Polished Split Layout
```
┌──────────────────┬──────────────────┐
│                  │  Gemma 4 Live    │
│                  │                  │
│    WEBCAM FEED   │  You: Hello!     │
│    (with glow    │  AI: Hi there!   │
│     border)      │                  │
│                  │  You: What's...  │
│                  │  AI: I can see...│
│                  │                  │
│                  │ ≋≋ waveform ≋≋   │
│                  │ [Cam] [Status]   │
└──────────────────┴──────────────────┘
```
- Evolution of current layout with visual polish
- Left: webcam with state-reactive border glow
- Right: modern chat with waveform at bottom
- **Pros**: Familiar, balanced, practical, full chat history
- **Cons**: Less visually dramatic, "just another chat app"

### D. "The Portal" — Centered Multimodal (CHOSEN)
```
┌─────────────────────────────────────┐
│           Gemma 4 Live    ● 2.1s    │
│                                     │
│        ┌─────────────────┐          │
│        │                 │          │
│        │   WEBCAM FEED   │ ← glow  │
│        │   (large, round │   border │
│        │    corners)     │   shows  │
│        │                 │   state  │
│        └─────────────────┘          │
│                                     │
│      ≋≋≋≋≋ audio waveform ≋≋≋≋≋    │
│                                     │
│   ┌─────────────────────────────┐   │
│   │ You: What do you see?       │   │
│   │ AI: I see you sitting at... │   │
│   └─────────────────────────────┘   │
│                                     │
│     [Camera]  ● Listening  [Mute]   │
└─────────────────────────────────────┘
```
- Centered vertical stack: webcam → waveform → transcript → controls
- Webcam border is the primary state indicator (animated gradient glow)
- Audio-reactive waveform between camera and transcript
- Compact transcript (latest exchange, not full history)
- **Pros**: Premium centered feel, camera prominent, clear state, responsive
- **Cons**: Slightly less space for each element

### E. "The Ambient" — Full-screen Gradient Mesh
```
┌─────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░  gradient mesh bg  ░░░░░░░░ │
│ ░░░░░  that breathes and  ░░░░░░░░ │
│ ░░░░  reacts to audio  ░░░░░░░░░░░ │
│ ░░░░░    ┌──────────┐     ░░░░░░░░ │
│ ░░░░░    │  WEBCAM   │    ░░░░░░░░ │
│ ░░░░░    │ (circle)  │    ░░░░░░░░ │
│ ░░░░░    └──────────┘     ░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░  "I see you at a desk"  ░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────────────────┘
```
- Entire background is audio-reactive gradient mesh
- Webcam in a circle, like a crystal ball
- Transcript floats over the gradient
- **Pros**: Most visually unique and impressive
- **Cons**: Complex to build, potentially distracting, readability concerns

## Decision: Alternative D "The Portal"

### Why This Design Wins

1. **Camera prominence**: The centered webcam is the hero — matching the multimodal differentiator. Users immediately see what makes this different from text chatbots.

2. **State communication**: The webcam border glow is large and unmissable. It wraps the primary visual element, so state is always in your peripheral vision.

3. **Audio waveform**: A canvas-based waveform between camera and transcript provides continuous visual feedback that the system is alive and responsive.

4. **Transcript grounding**: Compact but always visible. Users can verify what they said and what the AI heard.

5. **Centered layout**: Feels premium, "app-like", works on any screen size. Better than a left-right split for a demo.

6. **Buildable**: CSS animations + Canvas waveform. No Three.js or WebGL needed. Fits the single-file architecture.

7. **Demo-optimized**: Latency metrics visible but subtle. State transitions are dramatic enough to impress an audience.

### Visual Design Tokens
- Background: Deep dark (#050508) with subtle radial gradient
- State colors: Listening=#4ade80 (green), Processing=#f59e0b (amber), Speaking=#818cf8 (indigo)
- Border glow: 20px spread animated box-shadow matching state color
- Waveform: Canvas bars responding to audio analyser
- Typography: System font stack, size 14-16px for readability
- Webcam: 16:9, max-width 640px, border-radius 16px
- Transcript: Semi-transparent dark card, max 3-4 messages visible
