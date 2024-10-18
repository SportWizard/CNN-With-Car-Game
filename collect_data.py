import pyautogui
import keyboard
import pygetwindow
import os

def get_last_modified_file(directory, file_list):
    if not file_list:
        return None
    
    full_paths = [os.path.join(directory, f) for f in file_list]  # Construct full paths (e.g. outputs/forward/imgX.png)
    return max(full_paths, key=os.path.getmtime)  # Get the file with the most recent modification time

def extract_number_from_filename(filename):
    # Assuming filenames are in format "output/xxxx/imgX.png"
    start = filename.index("img") + len("img")
    end = filename.index(".")
    return int(filename[start:end])

def main():
    game_title = "art of rally"

    try:
        game_window = pygetwindow.getWindowsWithTitle(game_title)[0] # Get the first window with that title
        game_window.activate() # Put the window infront
        x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height

        x += 10
        y += 50
        width -= 20
        height -= 100

        f_dir = "outputs/forward"
        l_dir = "outputs/left"
        r_dir = "outputs/right"

        f_list = os.listdir(f_dir)
        l_list = os.listdir(l_dir)
        r_list = os.listdir(r_dir)

        f_last = get_last_modified_file(f_dir, f_list)
        l_last = get_last_modified_file(l_dir, l_list)
        r_last = get_last_modified_file(r_dir, r_list)

        # Set to 0 if the directory is empty else set it to the the last image's filename's number (e.g. imgX.png, set it to X)
        f_counter = 0 if not f_list else extract_number_from_filename(f_last)
        l_counter = 0 if not l_list else extract_number_from_filename(l_last)
        r_counter = 0 if not r_list else extract_number_from_filename(r_last)

        # Run the program until "esc" is pressed
        while True:
            # If one of these key are pressed, take a screenshot and put it in the corresponding directory
            if keyboard.is_pressed("up"):
                pyautogui.screenshot(imageFilename=f"outputs/forward/img{f_counter}.png", region=(x, y, width, height))
                print(f"outputs/forward/img{f_counter}.png captured")
                f_counter += 1

            if keyboard.is_pressed("left"):
                pyautogui.screenshot(imageFilename=f"outputs/left/img{l_counter}.png", region=(x, y, width, height))
                print(f"outputs/left/img{l_counter}.png captured")
                l_counter += 1

            if keyboard.is_pressed("right"):
                pyautogui.screenshot(imageFilename=f"outputs/right/img{r_counter}.png", region=(x, y, width, height))
                print(f"outputs/right/img{r_counter}.png captured")
                r_counter += 1

            # Stop the program
            if keyboard.is_pressed("esc"):
                print("Exiting...")
                break
    except:
        print(f"No window with title \"{game_title}\" found")

if __name__ == "__main__":
    main()