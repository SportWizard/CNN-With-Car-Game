from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import pyautogui
import pygetwindow
import keyboard
import time

def main():
    game_title = "art of rally"
    model = load_model("model.keras") # Load model

    game_window = pygetwindow.getWindowsWithTitle(game_title)[0] # Get the first window with that title
    game_window.activate() # Put the window infront
    x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height

    x += 10
    y += 50
    width -= 20
    height -= 100

    while True:
        img = pyautogui.screenshot(region=(x, y, width, height))
        img = np.array(img) # cv2 only read NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Colour correction

        img = cv2.resize(img, (64, 64))
        img = img_to_array(img)
        img = img / 255.0

        # Add a batch dimension: from (64, 64, 3) to (1, 64, 64, 3)
        img = np.expand_dims(img, axis=0)  # Now shape will be (1, 64, 64, 3)

        prediction = model.predict(img)

        # Get the index of the highest probability class
        predicted_class = np.argmax(prediction, axis=1)[0]

        if predicted_class == 0:
            keyboard.press("up")
            time.sleep(0.1)
            keyboard.release("up")
        elif predicted_class == 1:
            keyboard.press("left")
            time.sleep(0.3)
            keyboard.release("left")
        elif predicted_class == 2:
            keyboard.press("right")
            time.sleep(0.3)
            keyboard.release("right")

if __name__ == "__main__":
    main()