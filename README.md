<div align="center">

# 🔮 Aura: Vision Accessibility Core

### A 100% Offline, Privacy-First Assistive AI System for the Visually Impaired

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLOv11m-Object_Detection-00d4aa?style=flat-square)](https://docs.ultralytics.com)
[![llama3.2-vision](https://img.shields.io/badge/llama3.2--vision-VLM-8b5cf6?style=flat-square)](https://ollama.com/library/llama3.2-vision)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white?style=flat-square&logo=data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=)](https://ollama.com)
[![Flask](https://img.shields.io/badge/Flask-WebSocket-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

*Real-time spatial awareness, proximity warnings, and scene understanding — entirely on-device.*

---

</div>

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [The Problem](#-the-problem)
- [Dual-Tier Edge Architecture](#-dual-tier-edge-architecture)
- [Why Local Execution Matters](#-why-local-execution-matters)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Deep Dive](#-technical-deep-dive)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Executive Summary

**Aura** is an edge-deployed assistive AI system designed to serve as a **real-time spatial awareness tool** for visually impaired individuals. Unlike cloud-dependent solutions that introduce network latency and privacy risks, Aura executes **100% of its AI inference pipeline locally** on consumer hardware — specifically optimized for Apple Silicon MacBooks via Metal Performance Shaders (MPS).

The system provides three core capabilities:

| Capability | Model | Latency | Trigger |
|---|---|---|---|
| **Object Detection & Proximity Warnings** | YOLO11m | ~30ms per frame | Continuous (automatic) |
| **Scene Understanding & Navigation Context** | llama3.2-vision (via Ollama) | ~3-6s per query | Manual (user-initiated) |
| **Text Reading (OCR)** | EasyOCR | ~200ms per scan | Automatic (rate-limited) |

All outputs are delivered as **synthesized speech** via macOS TTS, with a **thread-safe audio queue** that prevents speech overlap — a critical UX requirement for users who depend entirely on auditory feedback.

---

## 🔍 The Problem

The World Health Organization estimates that **2.2 billion people** globally have a near or distance vision impairment. Existing assistive technology solutions suffer from three fundamental limitations:

1. **Cloud Dependency**: Services like Google Lookout and Microsoft Seeing AI require constant internet connectivity, making them unusable in elevators, subways, rural areas, or during network outages — precisely the environments where assistance is most needed.

2. **Privacy Violations**: Streaming continuous camera footage to cloud servers creates a persistent visual record of the user's private spaces, activities, and companions. For individuals who cannot visually verify what their camera captures, this represents an acute consent and dignity problem.

3. **Latency**: Cloud round-trip times of 200-500ms are acceptable for search queries but dangerous for spatial navigation. A 500ms delay in warning a user about an approaching obstacle can mean the difference between stopping safely and a collision.

**Aura eliminates all three problems** by moving the entire inference pipeline to the edge device.

---

## 🏗️ Dual-Tier Edge Architecture

Aura implements a **Dual-Tier Edge Architecture** that separates the inference pipeline into two cognitively distinct layers, each optimized for a different latency/depth tradeoff:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER'S MACBOOK (Edge Device)                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  TIER 1: REFLEX ENGINE                    Latency: ~30ms     │   │
│  │  ─────────────────────                                       │   │
│  │  Model:    YOLO11m (40MB, Apple MPS)                         │   │
│  │  Input:    640×480 camera frames @ 15 FPS                    │   │
│  │  Output:   Object labels + bounding boxes                    │   │
│  │  Logic:    Area-ratio proximity detection                    │   │
│  │  Audio:    "Caution. Chair immediately ahead."               │   │
│  │                                                               │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐  │   │
│  │  │  Camera      │───▶│  YOLO11m     │───▶│  Proximity Calc │  │   │
│  │  │  Capture     │    │  (MPS GPU)   │    │  (Area Ratio)   │  │   │
│  │  └─────────────┘    └──────────────┘    └────────┬────────┘  │   │
│  │                                                   │           │   │
│  │                                          ┌────────▼────────┐  │   │
│  │                                          │  Audio Queue    │  │   │
│  │                                          │  (TTS Output)   │  │   │
│  │                                          └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  TIER 2: COGNITIVE ENGINE                 Latency: ~3-6s     │   │
│  │  ─────────────────────                                       │   │
│  │  Model:    llama3.2-vision (11B params, via Ollama)          │   │
│  │  Input:    320×320 JPEG frame (manual trigger, downscaled)   │   │
│  │  Output:   1-sentence YOLO-grounded spatial description      │   │
│  │  Grounding: YOLO labels injected into prompt to prevent      │   │
│  │             hallucination (Contextual Grounding technique)   │   │
│  │  Timeout:  6s hard limit with YOLO-based audio fallback      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  INFRASTRUCTURE                                               │   │
│  │  ─────────────────                                            │   │
│  │  • EasyOCR:     Text reading (rate-limited to every 8s)      │   │
│  │  • Flask-SocketIO:  WebSocket Base64 video streaming         │   │
│  │  • SQLite:      Persistent environment logging               │   │
│  │  • Threading:   Audio queue (prevents TTS overlap)           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Two Tiers?

| Dimension | Tier 1 (Reflex) | Tier 2 (Cognitive) |
|---|---|---|
| **Analogy** | Peripheral vision — fast, instinctive | Focused attention — slow, deliberate |
| **Model** | YOLO11m (20M parameters) | llama3.2-vision (11B parameters) |
| **Latency** | ~30ms (real-time safe) | ~3-6s (acceptable for context) |
| **Trigger** | Continuous (automatic) | Manual (user button press) |
| **Output** | "Chair ahead" | "The chair is directly ahead about 3 feet away, with the desk to your left." |
| **GPU Memory** | ~500MB VRAM | ~6GB VRAM |

This separation is inspired by **Daniel Kahneman's Dual Process Theory** (System 1 / System 2 thinking), applied to the assistive AI domain:
- **Tier 1 (System 1)**: Fast, automatic, always-on object detection that runs reflexively
- **Tier 2 (System 2)**: Slow, deliberate scene comprehension that requires conscious user activation

---

## 🔒 Why Local Execution Matters

### 1. Privacy by Architecture

Unlike cloud-based assistive tools, Aura's privacy guarantee is **architectural, not policy-based**. No camera frame ever leaves the device — not because of a terms-of-service promise, but because the network stack is never invoked. This is a meaningful distinction for users who:

- Cannot visually verify what their camera is capturing
- May be in private medical, legal, or intimate settings
- Reside in jurisdictions with strict data sovereignty laws (GDPR, CCPA)

### 2. Zero Network Dependency

Aura functions identically whether the user has:
- Full 5G connectivity
- Spotty Wi-Fi in a basement
- No connectivity at all (airplane mode)

This is critical because the environments where visually impaired users need the most assistance — unfamiliar buildings, public transit, outdoor navigation — often have unreliable connectivity.

### 3. Deterministic Latency

Cloud inference latency is inherently variable (50ms–2000ms depending on server load, network conditions, and geographic distance). For an obstacle warning system, variable latency is worse than consistently high latency — because the user cannot develop reliable spatial intuition if the system's response time is unpredictable.

Aura's Tier 1 delivers **deterministic ~30ms inference** on Apple MPS, ensuring the user can trust the timing of proximity warnings.

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | macOS 13 (Ventura) | macOS 14+ (Sonoma) |
| **Chip** | Apple M1 | Apple M2 Pro / M3 |
| **RAM** | 8 GB | 16 GB |
| **Python** | 3.10 | 3.11+ |
| **Camera** | Built-in FaceTime | Any USB/built-in webcam |
| **Disk** | 5 GB free | 10 GB free |

> **Note**: An NVIDIA GPU with CUDA is not required. Aura is specifically optimized for Apple's Metal Performance Shaders (MPS) backend.

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/aura-vision-core.git
cd aura-vision-core
```

### Step 2: Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install & Configure Ollama (for Tier 2 — Cognitive Engine)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the llama3.2-vision multimodal model (~4.9 GB download)
ollama pull llama3.2-vision

# Start the Ollama server (runs on localhost:11434)
ollama serve
```

> **Important**: Ollama must be running before starting Aura if you want Tier 2 scene analysis. However, Aura will **degrade gracefully** — Tier 1 detection works perfectly without Ollama.

### Step 5: Launch Aura

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5001
```

---

## 🎮 Usage

| Action | Method | What Happens |
|---|---|---|
| **Start System** | Run `python app.py` | Camera activates, YOLO begins detection, TTS announces objects |
| **Proximity Warning** | Automatic | When a hazardous object fills >30% of the frame, an urgent audio warning is issued |
| **Scene Description** | Click "DESCRIBE SURROUNDINGS" | llama3.2-vision analyzes a YOLO-grounded, downscaled frame and speaks a 1-sentence spatial layout (6s hard timeout with YOLO fallback) |
| **Text Reading** | Automatic | EasyOCR scans for visible text every 8 seconds and displays it on the dashboard |
| **View Logs** | Dashboard bottom panel | SQLite-backed log of all scene analyses, persisted across sessions |

---

## 🔬 Technical Deep Dive

### Proximity Detection Algorithm

Rather than using computationally expensive monocular depth estimation, Aura implements a **bounding-box area ratio** heuristic:

```python
area_ratio = (box_width × box_height) / (frame_width × frame_height)

if area_ratio > 0.30:   # Object fills >30% of frame
    → EMERGENCY WARNING  # "Caution. {object} immediately ahead."
```

**Rationale**: As an object approaches a fixed camera, its projected bounding box area grows proportionally. A 30% area threshold corresponds roughly to an object within 1-2 meters — the critical distance for collision avoidance. This approach runs in O(1) time per detection, compared to O(n²) for depth estimation networks.

### WebSocket Video Streaming (vs. MJPEG / WebRTC)

| Approach | Pros | Cons |
|---|---|---|
| **MJPEG (HTTP streaming)** | Simple | Blocks 1 of 6 browser HTTP connections permanently |
| **WebRTC** | Low latency, P2P | Massive complexity (STUN/TURN/ICE), overkill for localhost |
| **WebSocket + Base64** ✅ | Non-blocking, bidirectional, simple | ~33% Base64 overhead |

Aura uses WebSocket Base64 streaming because:
1. The video feed is localhost-only, so Base64's 33% bandwidth overhead is irrelevant
2. The same WebSocket connection carries video, stats, VLM results, and logs — no connection pool exhaustion
3. WebRTC's NAT traversal infrastructure (STUN/TURN/ICE) adds zero value for a localhost application

### Thread-Safe Audio Queue

```
Main Thread (Flask) ──► Audio Queue (FIFO) ──► Audio Worker (Single Consumer)
                               │
                     Prevents TTS overlap
```

Without the queue, concurrent `os.system('say ...')` calls would produce garbled, simultaneous speech — catastrophic for a user who depends entirely on audio output. The single-consumer pattern guarantees sequential, intelligible utterance delivery.

---

## 📁 Project Structure

```
aura-vision-core/
├── app.py                  # Backend: Flask server, YOLO pipeline, VLM integration
├── templates/
│   └── index.html          # Frontend: WebSocket client, dashboard UI
├── instance/
│   └── vision_core.db      # SQLite database (auto-created)
├── yolo11m.pt              # YOLO11 Medium model weights (40 MB)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── venv/                   # Python virtual environment
```

---

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com)** — YOLO11 object detection framework
- **[Meta llama3.2-vision](https://ollama.com/library/llama3.2-vision)** — Multimodal vision-language model (air-gapped via Ollama)
- **[Ollama](https://ollama.com)** — Local LLM server infrastructure
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — Text detection and recognition
- **[Flask-SocketIO](https://flask-socketio.readthedocs.io)** — WebSocket transport layer

---

<div align="center">

*Built with the conviction that assistive AI should be private, fast, and available to everyone — regardless of connectivity.*

**Aura: Vision Accessibility Core** · MIT License

</div>
