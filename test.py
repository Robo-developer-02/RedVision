"""
============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by OpenAI
============================================================
  Stack:
    STT  → OpenAI Whisper (whisper-1)
    LLM  → OpenAI Chat model (streamed, sentence-by-sentence)
    TTS  → Microsoft Edge TTS (edge-tts, streamed + cached)
    Offline fallback → espeak (no internet required)

  Language Support:
    → Speak English → RoboBot replies & speaks in English
    → Speak Hindi   → RoboBot replies & speaks in Hindi
    → Switches instantly every message — no confusion

  Language Detection (3-layer):
    1. Whisper language tag  (fast, sometimes wrong)
    2. Script scan of transcript  (ground truth — never lies)
    3. Default → English

  Optimisations in this version:
    1. Streaming input  — speech is captured continuously and each
       sentence-sized chunk (separated by a short mid-turn pause) is
       transcribed in the background WHILE the user keeps talking.
    2. Streaming output — the LLM reply is streamed token-by-token;
       each finished sentence is sent to TTS and played immediately
       while the next sentence is still being generated/synthesised.
    3. Prompt caching   — (a) exact-match reply cache so a repeated
       question skips the LLM call entirely, (b) a disk-backed audio
       cache so a phrase is never re-synthesised twice, and (c) the
       system+history prefix is kept stable so OpenAI's own automatic
       prompt caching (models ≥1024 prompt tokens) can kick in.
    4. Empty strings are filtered out at every hand-off point before
       they ever reach an API (STT segment, combined transcript,
       individual streamed sentences).
    5/6/7. Layered, offline-safe error handling — see `handle_error()`.

  State Machine:
    IDLE ──(wake word)──► LISTENING ──(speech)──► THINKING ──► SPEAKING
      ▲                        │                                  │
      └──────(10s silence)─────┘◄─────────────────────────────────┘

  Wake word: "Hello" / "Hey"
============================================================
"""

import os
import re
import time
import queue
import socket
import hashlib
import tempfile
import threading
import subprocess
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import edge_tts
import pygame
from dotenv import load_dotenv

from openai import (
    OpenAI,
    APIError,          # base class — catches every OpenAI SDK error
                        # (APIConnectionError, APITimeoutError, RateLimitError,
                        #  AuthenticationError, APIStatusError, InternalServerError...)
)

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

STT_MODEL  = "whisper-1"
CHAT_MODEL = "gpt-4o"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 200

# ── VAD tuning ─────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.10
SILENCE_AFTER_SPEECH = 1.2   # seconds of silence → end of the whole turn
SENTENCE_PAUSE       = 0.45  # seconds of silence → mid-turn "sentence" break
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1

IDLE_TIMEOUT      = 10.0
IDLE_POLL_TIMEOUT = 30.0

WAKE_WORDS = ["hello", "hey"]

SYSTEM_EN = (
    "Your name is RoboBot. You are the helpful AI assistant . "
    "Keep responses concise and conversational."
    "No bullet points or markdown."
    "you are created by Robotwala."
    "dont use bullet points."
    "answer only in 2 to 3 sentences"
)

SYSTEM_HI = (
    "Aapka naam RoboBot hai. Aap helpful AI assistant hain. "
    "Apne uttar chhote aur batcheet ke andaz mein rakhein. "
    "Koi bullet points ya markdown nahi."
    "tumhein robotwala ne banaya hein."
    "bullet points ka use nahi karna he."
    "sirf 2 se 3 sentence me jawab dena he"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।\n])\s+")

# ── Fixed offline error strings (never sent to any network TTS) ──
MSG_NO_INTERNET = "Can't connect to the internet."
MSG_NO_SERVER   = "Can't connect to the server."
MSG_TRY_AGAIN   = "Please try again."

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

STATE_LABEL = {
    State.IDLE:      "😴 IDLE",
    State.LISTENING: "👂 LISTENING",
    State.THINKING:  "🤔 THINKING",
    State.SPEAKING:  "🔊 SPEAKING",
}


def show_state(state: State, note: str = ""):
    """Single always-visible line so the current state is obvious in the terminal."""
    line = f"\n[{STATE_LABEL[state]}]"
    if note:
        line += f" {note}"
    print(line)


# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)

# Separate history per language so the model never sees cross-language
# context and stays in the right language naturally. Kept as a STABLE
# ordered prefix (system + history) on every call so OpenAI's automatic
# prompt caching can match the repeated prefix.
history: dict = {"en": [], "hi": []}

# Exact-match reply cache: skips the LLM entirely for a repeated question.
reply_cache: dict = {}

pygame.mixer.init()

# Background workers: one pool for STT segments, one for TTS synthesis.
stt_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stt")
tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")


# ──────────────────────────────────────────────
#  CONNECTIVITY / ERROR HANDLING  (features 5, 6, 7)
# ──────────────────────────────────────────────

def is_internet_available(timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def speak_offline(text: str):
    """
    Offline, dependency-free TTS via the local `espeak` binary.
    Used ONLY for error announcements, since it never touches the
    network — unlike edge-tts, which would itself fail if the
    internet or the API is the actual problem.
    """
    print(f"   🔇 (offline) {text}")
    try:
        subprocess.run(
            ["espeak", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("   ⚠️  espeak is not installed — install it for spoken error messages "
              "(e.g. `sudo apt install espeak`).")


def handle_error(e: Exception, where: str):
    """
    Central error classifier. Order matters:
      1. No internet at all               → "Can't connect to the internet."
      2. Internet is fine but the OpenAI
         API call itself failed           → "Can't connect to the server."
      3. Anything else                    → "Please try again."
    """
    print(f"\n❌ Error in {where}: {type(e).__name__}: {e}")

    if not is_internet_available():
        speak_offline(MSG_NO_INTERNET)
        return

    if isinstance(e, APIError):
        speak_offline(MSG_NO_SERVER)
        return

    speak_offline(MSG_TRY_AGAIN)


# ──────────────────────────────────────────────
#  AUDIO CACHE  (feature 2 — part of "prompt caching")
# ──────────────────────────────────────────────

def _cache_path(text: str, voice: str) -> str:
    key = hashlib.sha256(f"{voice}::{text}".encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.mp3")


async def _tts_to_file(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def synthesize(text: str, voice: str) -> str:
    """
    Returns a path to an mp3 for `text`. Disk-cached by (voice, text) hash
    so any phrase — greetings, idle prompts, or a repeated LLM sentence —
    is only ever sent to edge-tts once.
    """
    path = _cache_path(text, voice)
    if os.path.exists(path):
        return path

    import asyncio
    tmp_path = path + ".tmp"
    asyncio.run(_tts_to_file(text, tmp_path, voice))
    os.replace(tmp_path, path)
    return path


def play(path: str):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)
    pygame.mixer.music.unload()


# ──────────────────────────────────────────────
#  VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


# ──────────────────────────────────────────────
#  STREAMING SPEAK  — plays sentences as they arrive (feature 1, output half)
# ──────────────────────────────────────────────

class StreamingSpeaker:
    """
    Consumer that plays synthesised sentences in order while the producer
    (LLM stream) is still generating later sentences. Synthesis for
    sentence N+1 happens in the background while sentence N is playing.
    """

    def __init__(self, lang: str):
        self.lang = lang
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        self._announced = False

    def _consume(self):
        while True:
            future = self._q.get()
            if future is None:
                break
            path = future.result()
            play(path)

    def say(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:                       # feature 4 — never pass empty text
            return
        if not self._announced:
            show_state(State.SPEAKING)
            self._announced = True
        print(f"   💬 {sentence}")
        voice = pick_voice(sentence, self.lang)
        future = tts_executor.submit(synthesize, sentence, voice)
        self._q.put(future)

    def finish(self):
        self._q.put(None)
        self._thread.join()


def speak_blocking(text: str, lang: str = "en"):
    """Simple one-shot speak for short fixed prompts (greeting, idle, wake-ack)."""
    text = text.strip()
    if not text:
        return
    voice = pick_voice(text, lang)
    path = synthesize(text, voice)
    play(path)


# ──────────────────────────────────────────────
#  VAD RECORDING WITH MID-TURN SENTENCE SEGMENTATION  (feature 1, input half)
# ──────────────────────────────────────────────

def transcribe_segment(audio: np.ndarray) -> Tuple[str, str]:
    """Whisper call for a single audio segment. Returns (text, lang)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
                response_format="verbose_json",
            )
    finally:
        os.unlink(tmp_path)

    text = (result.text or "").strip()
    lang = (result.language or "en").strip().lower()
    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    return text, lang


def capture_and_transcribe(timeout: float) -> Tuple[Optional[str], str]:
    """
    Records one full user turn via VAD. Whenever a short mid-turn pause
    (SENTENCE_PAUSE) is detected, the audio collected so far is sent off
    for transcription in the background immediately — while the user is
    still talking — instead of waiting for the whole turn to end.

    Returns (combined_text_or_None, lang).
    """
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
        blocksize=blocksize, callback=callback,
    )
    stream.start()

    pre_buffer: list = []
    segment_buffer: list = []
    recording = False
    silence_start: Optional[float] = None
    segment_split_done = False
    idle_clock = time.time()
    futures: List = []

    def flush_segment():
        nonlocal segment_buffer
        if not segment_buffer:
            return
        audio = np.concatenate(segment_buffer, axis=0)
        segment_buffer = []
        if len(audio) < SAMPLE_RATE * 0.15:   # too short to bother
            return
        futures.append(stt_executor.submit(transcribe_segment, audio))
        print("   ✂️  segment captured → transcribing in background...")

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    stream.stop(); stream.close()
                    return None, "en"
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock = time.time()
                silence_start = None
                segment_split_done = False
                if not recording:
                    recording = True
                    segment_buffer = list(pre_buffer)
                segment_buffer.append(chunk)

            elif recording:
                segment_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                    continue
                elapsed = time.time() - silence_start
                if elapsed >= SILENCE_AFTER_SPEECH:
                    break  # end of the whole turn
                if elapsed >= SENTENCE_PAUSE and not segment_split_done:
                    segment_split_done = True
                    flush_segment()

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    stream.stop(); stream.close()
                    return None, "en"
    finally:
        stream.stop()
        stream.close()

    flush_segment()  # final trailing segment

    if not futures:
        return None, "en"

    pieces = []
    lang = "en"
    for fut in futures:
        try:
            text, seg_lang = fut.result()
        except Exception:
            continue
        if text:
            pieces.append(text)
            lang = seg_lang  # last non-empty segment's script-scan wins

    combined = " ".join(p for p in pieces if p.strip()).strip()

    # Re-run the script scan over the FULL combined text so language never
    # flips mid-sentence just because one short segment mis-detected.
    for ch in combined:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    if not combined or len(combined) < 1:
        return None, lang

    return combined, lang


# ──────────────────────────────────────────────
#  WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ──────────────────────────────────────────────
#  AI REPLY — streamed, sentence-by-sentence, with reply caching
# ──────────────────────────────────────────────

def stream_ai_reply_and_speak(user_text: str, lang: str) -> str:
    """
    Streams the LLM reply and speaks it sentence-by-sentence as it's
    generated. Returns the full reply text (for history bookkeeping,
    which already happened inline).
    """
    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN
    lang_history = history[lang]

    cache_key = (lang, user_text.strip().lower())
    speaker = StreamingSpeaker(lang)

    # ── Prompt cache hit: skip the LLM call entirely ─────────────
    if cache_key in reply_cache:
        print("   💾 cache hit — skipping LLM call")
        cached_reply = reply_cache[cache_key]
        lang_history.append({"role": "user", "content": user_text})
        lang_history.append({"role": "assistant", "content": cached_reply})
        for sentence in SENTENCE_SPLIT_RE.split(cached_reply):
            speaker.say(sentence)
        speaker.finish()
        return cached_reply

    # ── Stable prefix (system + history) so OpenAI's own automatic
    #    prompt caching can match repeated prefixes across turns ──
    lang_history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": system}, *lang_history]

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        stream=True,
    )

    buffer = ""
    full_reply = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:                     # feature 4 — skip empty deltas
            continue
        buffer += delta
        full_reply += delta

        parts = SENTENCE_SPLIT_RE.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                speaker.say(sentence)
            buffer = parts[-1]

    if buffer.strip():
        speaker.say(buffer)

    speaker.finish()

    full_reply = full_reply.strip()
    if full_reply:                        # feature 4 — don't cache/store empties
        lang_history.append({"role": "assistant", "content": full_reply})
        reply_cache[cache_key] = full_reply

    return full_reply


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 56)
    print("  🤖 RoboBot — streaming speech-to-speech assistant")
    print("=" * 56)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' to wake up")
    print("    🤔 THINKING   — waiting on the first tokens")
    print("    🔊 SPEAKING   — plays each sentence as it's ready")
    print("  Ctrl+C to quit")
    print("=" * 56 + "\n")


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print_banner()

    state = State.LISTENING
    lang  = "hi"

    try:
        speak_blocking(
            "Hello, mein RoboBot hoon, mein aapki kese madad kar sakta hoon, "
            "Krapya apna sawaal poochhiye .",
            lang="hi",
        )
    except Exception as e:
        handle_error(e, "opening greeting")

    try:
        while True:

            # ════════════════════════ IDLE ════════════════════════
            if state == State.IDLE:
                show_state(state, "— say 'Hello' to activate...")
                try:
                    combined, _ = capture_and_transcribe(timeout=IDLE_POLL_TIMEOUT)
                except Exception as e:
                    handle_error(e, "idle listening")
                    continue

                if not combined:
                    continue

                print(f"   Heard: {combined!r}")
                if is_wake_word(combined):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    try:
                        speak_blocking("Haan, mein sun raha hoon. Aap apna sawaal poochhiye.", lang="hi")
                    except Exception as e:
                        handle_error(e, "wake acknowledgement")
                else:
                    print("   Not a wake word — staying idle.")
                continue

            # ═════════════════════ LISTENING ═══════════════════════
            if state == State.LISTENING:
                show_state(state, f"— silence for {int(IDLE_TIMEOUT)}s → idle")
                try:
                    user_text, lang = capture_and_transcribe(timeout=IDLE_TIMEOUT)
                except Exception as e:
                    handle_error(e, "listening / transcription")
                    continue

                if user_text is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    try:
                        speak_blocking(
                            "Mein abhi idle mode mein ja raha hoon. Jab zaroorat ho, 'Hello' kahiye.",
                            lang="hi",
                        )
                    except Exception as e:
                        handle_error(e, "idle announcement")
                    continue

                user_text = user_text.strip()
                if not user_text:             # feature 4 — never forward empty text
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")
                state = State.THINKING
                continue

            # ═════════════════════ THINKING ════════════════════════
            if state == State.THINKING:
                show_state(state)
                try:
                    stream_ai_reply_and_speak(user_text, lang)
                except Exception as e:
                    handle_error(e, "LLM reply / speech synthesis")
                    state = State.LISTENING
                    continue
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        stt_executor.shutdown(wait=False, cancel_futures=True)
        tts_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()