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
    os.system("clear")
    print("=================================")
    print("        ROBOT CONTROLLER")
    print("=================================")
    print("0 -> Chatbot")
    print("1 -> Audio Player")
    
    print("2 -> Exit (You will enter the Raspberry Pi Terminal)")
    print("IF YOU HAVE ENTERED THE TERMINAL , paste these commands in order -")
    print("cd Desktop/CHATBOT/")
    print("python launcher.py")
# --------------------------------------------------------------------
# Main loop: show the menu, run the chosen program, wait for it to
# finish, then show the menu again. Repeats until the user exits.
# --------------------------------------------------------------------
try:
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            # Launch Chatbot
            subprocess.run([sys.executable, CHATBOT_PATH])

        elif choice == "1":
            # Launch Audio Player
            subprocess.run([sys.executable, AUDIO_PLAYER_PATH])

        elif choice == "2":
            # Return to the Raspberry Pi terminal
            print("Returning to Raspberry Pi terminal...")
            break

        elif choice == "3":
            # Exit completely
            print("Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice.")
            input("Press Enter to continue...")

except KeyboardInterrupt:
    # Ctrl+C - exit cleanly instead of showing a traceback.
    print("\nInterrupted. Exiting Launcher. Goodbye!")
