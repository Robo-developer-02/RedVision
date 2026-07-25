"""
audio_player.py

A simple command-line audio player for Raspberry Pi 4.
Plays .m4a files stored in the audio/ folder.

No classes, no threading you have to manage, no queues, no config
files - just plain variables, a couple of small functions, and a
while loop that reads commands from the terminal.

Playback is handled by VLC (via the python-vlc bindings). VLC plays
audio only - it never touches the microphone.
"""

from pathlib import Path
import vlc

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------

# Folder that holds all the audio files, next to this script.
AUDIO_DIR = Path(__file__).resolve().parent / "audio"

# One VLC instance and one media player, created once and reused for
# every track. `player` is the object STOP calls .stop() on.
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()


def on_playback_end(event):
    """
    VLC calls this by itself the moment a track finishes on its own
    (this is VLC's internal mechanism, not something we build - we
    just tell it which function to call). It lets us print
    "Playback Finished." without blocking the command loop while
    waiting for the track to end, which would stop STOP from working
    immediately.
    """
    print("Playback Finished.")


# Tell VLC to call on_playback_end() when a track reaches its end.
player.event_manager().event_attach(vlc.EventType.MediaPlayerEndReached, on_playback_end)


# --------------------------------------------------------------------
# Playback helper
# --------------------------------------------------------------------

def play_audio(filename):
    """Load and play one audio file from the audio/ folder."""
    file_path = AUDIO_DIR / filename

    if not file_path.exists():
        print(f"Error: file not found -> {file_path}")
        return

    media = vlc_instance.media_new(str(file_path))
    player.set_media(media)
    player.play()
    print(f"Playing: {filename}")


# --------------------------------------------------------------------
# Help text
# --------------------------------------------------------------------

def print_help():
    """Print all supported commands."""
    print("Available Commands:")
    print("  1      - Play welcome.m4a")
    print("  2      - Play intro.m4a")
    print("  3      - Play announcement.m4a")
    print("  4      - Play goodbye.m4a")
    print("  STOP   - Stop the currently playing audio")
    print("  HELP   - Show this help message")
    print("  EXIT   - Exit the program")


# --------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------

print("Audio Player Ready. Type HELP for a list of commands.")

# To add a 5th audio file: drop the .m4a file in audio/, then add
#     elif command == "5":
#         play_audio("your_file.m4a")
# below, next to the existing commands.

try:
    while True:
        command = input("Command > ").strip().upper()

        if command == "1":
            play_audio("welcome.m4a")
        elif command == "2":
            play_audio("intro.m4a")
        elif command == "3":
            play_audio("announcement.m4a")
        elif command == "4":
            play_audio("goodbye.m4a")
        elif command == "STOP":
            player.stop()
            print("Stopped.")
        elif command == "HELP":
            print_help()
        elif command == "EXIT":
            player.stop()
            print("Exiting Audio Player. Goodbye!")
            break
        else:
            print("Unknown command.")

except KeyboardInterrupt:
    # Ctrl+C - stop cleanly instead of showing an error traceback.
    player.stop()
    print("\nInterrupted. Exiting Audio Player. Goodbye!")
