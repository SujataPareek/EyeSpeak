# 🛡️ Aura — Defense Q&A Preparation

## The 3 Hardest Questions an Evaluator Will Ask

---

## Question 1: "Why WebSockets instead of WebRTC for the video stream?"

### The Scripted Answer:

> "That's a great question, and it was a deliberate architectural decision.
>
> WebRTC was designed to solve a very specific problem: **real-time peer-to-peer media transport across NAT boundaries over the public internet.** To do this, it requires STUN servers to discover public IP addresses, TURN servers as relay fallbacks, and the ICE protocol to negotiate the optimal connection path. This is a massive amount of infrastructure complexity.
>
> In Aura, the camera feed travels from a Python backend to a browser **on the same machine** — `localhost:5001`. There is no NAT traversal, no public internet, no peer discovery needed. Using WebRTC here would be like hiring an international shipping company to move a box from your kitchen to your living room.
>
> I initially considered MJPEG streaming via a standard HTTP endpoint — which is what many OpenCV tutorials suggest. But MJPEG has a critical flaw: **it permanently occupies one of the browser's 6 per-domain HTTP/1.1 connections.** This means if I'm streaming video over MJPEG, and the browser also needs connections for XHR requests, favicon loading, and other resources, I've already consumed 1 of those 6 slots indefinitely. That's a 16.7% reduction in connection capacity for a simple video feed.
>
> **WebSocket Base64 streaming solves both problems:** it uses a single persistent connection that carries video frames, object detection stats, VLM results, OCR text, and database logs — all multiplexed over one channel. Yes, Base64 encoding adds a 33% size overhead compared to raw binary. But at 640×480 JPEG quality 50, each frame is approximately 15KB — so the Base64 overhead is about 5KB per frame. At 15 FPS, that's 75KB/s of overhead on a localhost loopback interface that supports gigabits per second. The overhead is literally unmeasurable.
>
> If this were a production system deployed on a network — say, a Raspberry Pi streaming to a phone — I'd reconsider. But for a localhost assistive tool, WebSocket Base64 is the **optimal tradeoff** of simplicity, multiplexing, and performance."

### Why This Answer Scores Top Marks:
- Shows you **considered alternatives** (MJPEG, WebRTC)
- Explains **why each was rejected** with specific technical reasons
- Quantifies the Base64 overhead (15KB → 20KB per frame)
- Demonstrates awareness of browser connection pooling limits
- Acknowledges the architecture would change for different deployment scenarios

---

## Question 2: "How do you handle VLM hallucinations? What if llama3.2-vision describes objects that aren't there?"

### The Scripted Answer:

> "VLM hallucination is a fundamental challenge in the field — it's not a bug specific to any one model, but an inherent limitation of autoregressive language models. Here's how Aura mitigates this at **four levels**, three of which are implemented:
>
> **Level 1 — Contextual Grounding (Implemented).** This is our **primary anti-hallucination technique.** Instead of asking the VLM 'what do you see?' — which invites fabrication — we inject the YOLO-confirmed object list directly into the prompt. The prompt architecture is: *'I already know this scene contains a chair and a person. Describe the spatial layout and immediate hazards in one short sentence. Do not list the items.'* This constrains the VLM to describing **spatial relationships between known objects**, not identifying objects from scratch. We're offloading the 'what' to YOLO (which is a discriminative model that cannot hallucinate) and only asking the VLM for the 'where.'
>
> **Level 2 — Architectural Separation.** Aura's Tier 1 Reflex Engine — the YOLO detector — runs independently and continuously. It's a discriminative model that outputs **only what it was trained to detect**, with confidence scores. The VLM (Tier 2) is deliberately isolated as a **manual-trigger-only supplement.** If llama3.2-vision says 'there's a desk to your left' and there isn't one, the cost is minor inconvenience. But if YOLO misses an obstacle, the cost is a collision. Safety-critical hazard warnings come exclusively from Tier 1.
>
> **Level 3 — Output Constraint.** I set `temperature: 0.1` and `num_predict: 45` in the Ollama payload. The near-zero temperature forces deterministic, factual output — higher values (0.7+) introduce creative variation that is indistinguishable from hallucination in safety contexts. The 45-token cap forces a single-sentence response, minimizing drift from visual grounding. Hallucinations correlate strongly with output length — the longer a model generates, the more it diverges from the image.
>
> **Level 4 — Cross-Validation (Potential Future Work).** We could check the VLM's response against the concurrent YOLO detections — if the VLM mentions an object YOLO hasn't seen, that sentence could be suppressed. This is architecturally trivial since both data streams are available, but I haven't implemented it yet to keep the system simple for this version.
>
> The key insight is that **Contextual Grounding + the dual-tier architecture together form the hallucination mitigation strategy.** By design, no safety-critical decision depends on the VLM."

### Why This Answer Scores Top Marks:
- Acknowledges the problem is fundamental, not dismissable
- Shows **Contextual Grounding is implemented**, not theoretical
- Explains `temperature: 0.1` and `num_predict: 45` as concrete technical mitigations
- Proposes a credible future improvement (cross-validation)
- Ties back to the dual-tier design as the primary defense

---

## Question 3: "Your proximity detection uses bounding box area ratio. That's a rough heuristic — why not use monocular depth estimation?"

### The Scripted Answer:

> "You're right that bounding box area ratio is a heuristic — it's not true depth estimation. But I chose it deliberately after evaluating the tradeoffs for this specific use case.
>
> **Monocular depth estimation** — using a model like MiDaS or Depth Anything — would give me per-pixel depth maps. But there are three problems:
>
> **First, computational cost.** MiDaS v3.1 Large processes a single 384×384 frame in approximately 100-150ms on Apple MPS. My current YOLO11m inference runs in ~30ms. Adding depth estimation would increase per-frame processing time by 4-5x, dropping my effective detection rate from 15 FPS to approximately 3-4 FPS. For an obstacle warning system, frame rate IS safety. Missing a frame because we're computing depth is worse than having an approximate depth from a fast heuristic.
>
> **Second, integration complexity.** Depth maps give me depth at every pixel, but I still need to associate that depth with specific objects. That means I'd need to: run YOLO for detection, run MiDaS for depth, then compute the median depth within each YOLO bounding box. The bounding box area ratio gives me an analogous signal — 'how close is this object?' — without the second inference pass.
>
> **Third, and most importantly: for the warning use case, precision doesn't matter — only recall does.** I don't need to know that a chair is exactly 1.3 meters away. I need to know that a chair is 'dangerously close' versus 'safely distant.' The 30% area threshold cleanly separates these two categories for objects at typical indoor scales. A chair filling 30% of a 640×480 frame corresponds roughly to an object within 1-2 meters — which is exactly the warning distance I need.
>
> That said, if I were building this for outdoor navigation where depth precision matters more — say, detecting a curb exactly at the foot boundary — I'd incorporate a lightweight depth model like Depth Anything Small and accept the FPS penalty. The architecture is designed so that swapping the proximity calculation is a **single-function change** — the rest of the pipeline (audio queue, WebSocket emission, logging) is completely decoupled."

### Why This Answer Scores Top Marks:
- Demonstrates you **considered and rejected** the sophisticated approach
- Quantifies the performance penalty (100-150ms vs 30ms)
- Distinguishes between precision and recall requirements
- Shows awareness of state-of-the-art models (MiDaS, Depth Anything)
- Proposes a concrete scenario where you'd change your approach
- Highlights the **modular architecture** allows future swaps

---

## Bonus: Quick-Fire Questions & One-Line Answers

| Question | Answer |
|---|---|
| *"Why Flask instead of FastAPI?"* | "Flask-SocketIO has mature WebSocket support with threading mode. FastAPI's WebSocket support would require me to rewrite the camera loop as an async generator and manage asyncio event loops — unnecessary complexity for a single-user localhost tool." |
| *"Why SQLite instead of PostgreSQL?"* | "SQLite is serverless — zero configuration, zero processes. Aura is a single-user edge application, not a multi-tenant service. SQLite handles concurrent reads and single-writer workloads perfectly, and the entire database is a single portable file." |
| *"Isn't `os.system('say ...')` a security risk?"* | "Yes, raw string interpolation into shell commands is a classic injection vector. That's why I sanitize the text by escaping single quotes before passing it to `os.system`. In a production system, I'd use `subprocess.run(['say', text])` which avoids the shell entirely. For this prototype, the input sources are model outputs and hardcoded labels — both controlled." |
| *"What happens if two users access the dashboard?"* | "The WebSocket broadcasts to all connected clients, so both see the same feed. The camera is a shared resource — this is a single-user assistive tool by design, not a multi-tenant service." |
| *"Why not use browser's Web Speech API instead of macOS `say`?"* | "Two reasons: First, server-side TTS means audio plays even if the browser tab is backgrounded or closed — critical for an assistive tool. Second, the audio queue pattern guarantees sequential delivery across all event sources (YOLO, VLM, OCR). Browser TTS would require complex client-side queueing with the SpeechSynthesis API's own event model." |

---

> **Pro Tip**: When answering, always use the structure: **"I considered X, but chose Y because Z."** This demonstrates engineering judgment, not just implementation skill.
