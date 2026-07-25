# Robot Controller

A Raspberry Pi based robot controller that provides a simple terminal menu to launch either an AI Voice Chatbot or an Audio Player. The project is designed to be controlled remotely over SSH using the Termux application on Android.

---

## Features

### Robot Controller (Launcher)

- Interactive terminal menu.
- Automatically launches after SSH login.
- Runs inside the project's Python virtual environment.
- Returns to the main menu after the selected program exits.
- Beginner-friendly and lightweight implementation.

Menu:

```
=================================
        ROBOT CONTROLLER
=================================

0 -> Chatbot
1 -> Audio Player
2 -> Exit
```

---

## AI Voice Chatbot

The chatbot is a voice-based conversational assistant developed for Raspberry Pi.

### Features

- Speech-to-Text (STT)
- Text-to-Speech (TTS)
- Natural language conversations
- Supports voice interaction
- Runs completely from the terminal
- Uses the project's virtual environment
- Returns to the launcher when the chatbot exits

Chatbot file:

```
test.py
```

---

## Audio Player

The audio player is a lightweight command-line application for playing pre-recorded audio announcements.

### Features

- Plays `.m4a` audio files
- Speaker output only (no microphone access)
- Controlled directly from the terminal
- Simple command-based interface
- Easy to extend by adding more audio files

Supported commands:

```
1
2
3
4
HELP
STOP
EXIT
```

Example:

```
Command > 1

Playing: welcome.m4a

Playback Finished.
```

Audio Player location:

```
audio_player/
```

---

## Project Structure

```
CHATBOT/
│
├── launcher.py
├── test.py
│
├── audio_player/
│   ├── audio_player.py
│   ├── requirements.txt
│   └── audio/
│       ├── welcome.m4a
│       ├── intro.m4a
│       ├── announcement.m4a
│       └── goodbye.m4a
│
├── venv/
└── README.md
```

---

## Virtual Environment

The project uses its own Python virtual environment.

Location:

```
/home/lucy/Desktop/CHATBOT/venv
```

All programs launched through `launcher.py` automatically use this virtual environment.

---

## Running the Project

SSH into the Raspberry Pi:

```bash
ssh lucy@<RaspberryPi-IP>
```

After login, the Robot Controller menu starts automatically.

Choose:

```
0 -> AI Voice Chatbot
```

or

```
1 -> Audio Player
```

When either program exits, control returns to the launcher menu.

---

## Audio Files

Store all announcement files inside:

```
audio_player/audio/
```

Example:

```
audio/
├── welcome.m4a
├── intro.m4a
├── announcement.m4a
└── goodbye.m4a
```

To add a new announcement:

1. Copy the `.m4a` file into the `audio` folder.
2. Add a new command inside `audio_player.py`.

Example:

```python
elif command == "5":
    play_audio("new_audio.m4a")
```

---

## Technologies Used

- Python 3
- Raspberry Pi OS
- SSH
- Termux
- Virtual Environment (venv)
- Speech-to-Text
- Text-to-Speech
- VLC (Audio Playback)

---

## Workflow

```
Raspberry Pi Boot
        │
        ▼
SSH Login (Termux)
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

## Notes

- The launcher always uses the project's virtual environment.
- The chatbot and audio player are completely independent applications.
- The audio player never accesses the microphone.
- The chatbot and audio player are never run simultaneously.
- All interaction is performed through an SSH terminal.

---

## Future Improvements

- GPIO control integration
- Robot movement commands
- Camera integration
- Remote command interface
- Autonomous operating modes
- Additional audio announcements
