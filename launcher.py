"""
launcher.py

A simple terminal menu that launches either the chatbot or the
audio player, waits for it to finish, then shows the menu again.

No classes, no threading - just plain variables, a couple of
subprocess calls, and a while loop.
"""

import os
import subprocess
import sys

# --------------------------------------------------------------------
# Fixed paths to the two programs this launcher can start.
# Change these two lines if you ever move the files.
# --------------------------------------------------------------------
CHATBOT_PATH = "/home/lucy/Desktop/CHATBOT/test.py"
AUDIO_PLAYER_PATH = "/home/lucy/Desktop/CHATBOT/audio_player/audio_player.py"


def show_menu():
    """Clear the screen and print the menu."""
    os.system("clear")
    print("=================================")
    print("        ROBOT CONTROLLER")
    print("=================================")
    print("0 -> Chatbot")
    print("1 -> Audio Player")
    print("2 -> Exit")


# --------------------------------------------------------------------
# Main loop: show the menu, run the chosen program, wait for it to
# finish, then show the menu again. Repeats until the user exits.
# --------------------------------------------------------------------
try:
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            # subprocess.run() waits here until test.py exits on its own,
            # then control comes back to the launcher automatically.
            subprocess.run([sys.executable, CHATBOT_PATH])
        elif choice == "1":
            subprocess.run([sys.executable, AUDIO_PLAYER_PATH])
        elif choice == "2":
            print("Exiting Launcher. Goodbye!")
            break
        else:
            print("Invalid choice.")
            input("Press Enter to continue...")

except KeyboardInterrupt:
    # Ctrl+C - exit cleanly instead of showing a traceback.
    print("\nInterrupted. Exiting Launcher. Goodbye!")