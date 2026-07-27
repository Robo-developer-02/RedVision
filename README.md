# 🤖 Robot Controller

A Raspberry Pi based robot controller that provides a simple terminal interface to launch either an AI Voice Chatbot or an Audio Player. The entire project is designed to be operated remotely over SSH (using applications like Termux) and runs inside a dedicated Python virtual environment.

---

# Features

## Robot Controller (Launcher)

- Simple terminal-based menu
- Automatically starts after SSH login
- Launches the AI Chatbot or Audio Player
- Returns to the launcher after exiting either application
- Lightweight and beginner-friendly implementation

```
=================================
        ROBOT CONTROLLER
=================================

0 -> Chatbot
1 -> Audio Player
2 -> Exit
```

---

# AI Voice Chatbot

The chatbot is a real-time speech-to-speech conversational assistant designed for Raspberry Pi.

## Features

### Speech Processing

- OpenAI Whisper Speech-to-Text
- Microsoft Edge Neural Text-to-Speech
- Offline speech fallback using espeak
- Continuous voice conversation
- Wake-word activation
- Automatic silence detection

### AI

- GPT-4o powered conversations
- Streaming AI responses
- Short conversational replies
- English and Hindi support
- Automatic language detection
- Separate conversation history for each language

### Performance Optimizations

- Streaming Speech-to-Text
- Streaming GPT output
- Streaming Text-to-Speech playback
- Prompt caching
- Audio caching
- Background transcription
- Background speech synthesis
- Low latency response pipeline

### Error Handling

- Internet connectivity detection
- OpenAI API error handling
- Offline fallback speech
- Automatic recovery from errors

### State Machine

```
             Hello / Hey
                 │
                 ▼
           LISTENING
                 │
                 ▼
           THINKING
                 │
                 ▼
            SPEAKING
                 │
                 ▼
           LISTENING

      10 seconds silence
                 │
                 ▼
               IDLE
                 │
       Say "Hello" again
                 │
                 └────────► LISTENING
```

---

# Audio Player

A lightweight command-line audio player that plays pre-recorded announcements using VLC.

## Features

- Plays `.m4a` audio files
- Uses VLC backend
- Speaker output only
- Never accesses the microphone
- Non-blocking playback
- Stop playback anytime
- Easily extendable with additional audio files

Supported Commands

```
1
2
3
4
HELP
STOP
EXIT
```

Example

```
Command > 1

Playing: welcome.m4a

Playback Finished.
```

---

# Project Structure

```
CHATBOT/
│
├── launcher.py
├── test.py
├── README.md
├── requirements.txt
├── .env
│
├── tts_cache/
│
├── audio_player/
│   ├── audio_player.py
│   └── audio/
│       ├── welcome.m4a
│       ├── intro.m4a
│       ├── announcement.m4a
│       └── goodbye.m4a
│
└── venv/
```

---

# Technologies Used

| Component | Technology |
|------------|------------|
| Programming Language | Python 3 |
| Hardware | Raspberry Pi 4 |
| Speech Recognition | OpenAI Whisper |
| AI Model | GPT-4o |
| Text-to-Speech | Microsoft Edge TTS |
| Offline Speech | espeak |
| Audio Playback | pygame |
| Announcement Playback | VLC |
| Audio Streaming | mpg123 |
| Audio Input | sounddevice |
| Audio Processing | soundfile |
| Environment Variables | python-dotenv |
| SSH Access | OpenSSH |
| Remote Terminal | Termux |

---

# Python Requirements

Install the required Python packages inside your virtual environment.

```bash
python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install \
openai \
numpy \
sounddevice \
soundfile \
edge-tts \
pygame \
python-dotenv \
python-vlc
```

Or simply install everything using

```bash
pip install -r requirements.txt
```

---

# requirements.txt

```
edge-tts
numpy
openai
pygame
python-dotenv
python-vlc
sounddevice
soundfile
```

---

# System Dependencies

Install the required Linux packages.

```bash
sudo apt update

sudo apt install \
python3-dev \
portaudio19-dev \
libportaudio2 \
ffmpeg \
espeak \
mpg123 \
vlc \
libvlc-dev
```

---

# Environment Variables

Create a `.env` file in the project directory.

```
OPENAI_API_KEY=your_openai_api_key
```

---

# Running the Project

Activate the virtual environment

```bash
source venv/bin/activate
```

Run the launcher

```bash
python launcher.py
```

Or, if configured, simply SSH into the Raspberry Pi.

```bash
ssh <username>@<raspberry-pi-ip>
```

The Robot Controller menu will automatically appear.

---

# Audio Files

Store all announcement files inside

```
audio_player/audio/
```

Example

```
audio/
├── welcome.m4a
├── intro.m4a
├── announcement.m4a
└── goodbye.m4a
```

To add a new announcement

1. Copy the audio file into the `audio` directory.
2. Add a new command inside `audio_player.py`.

Example

```python
elif command == "5":
    play_audio("new_audio.m4a")
```

---

# Audio Pipeline

```
Microphone
      │
      ▼
SoundDevice
      │
      ▼
OpenAI Whisper
      │
      ▼
GPT-4o
      │
      ▼
Microsoft Edge TTS
      │
      ▼
pygame
      │
      ▼
Speaker
```

---

# Launcher Workflow

```
Raspberry Pi Boot
        │
        ▼
SSH Login
        │
        ▼
launcher.py
        │
        ├───────────────┐
        ▼               ▼
 AI Voice Chatbot   Audio Player
        │               │
        └───────┬───────┘
                ▼
        Robot Controller Menu
```

---

# Notes

- Built specifically for Raspberry Pi.
- Runs entirely from the terminal.
- Chatbot and Audio Player are independent applications.
- Only one application runs at a time.
- Audio Player never accesses the microphone.
- Chatbot automatically switches between English and Hindi.
- Uses streaming responses for lower latency.
- Includes prompt and audio caching for improved performance.
- Gracefully handles network failures and API errors.

---

# Future Improvements

- GPIO robot movement
- ESP32 integration
- Camera support
- Face recognition
- Object detection
- Autonomous navigation
- ROS integration
- Battery monitoring
- Web dashboard
- Multi-language support
- Robot arm control

---

# License

This project is intended for educational, research, and robotics development purposes.

---

# Author

Developed by **Robotwala**
