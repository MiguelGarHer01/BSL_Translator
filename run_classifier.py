from mode import gesture_mode
import keyboard

while True:

    if keyboard.is_pressed('e'):
        gesture_mode()

    if keyboard.is_pressed('s'):
        print("SPELL")