import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import imageio


def load_image(file_path):
    """Load image and convert it to RGB for displaying in Tkinter."""
    return imageio.imread(file_path)


def stitch_images():
    if len(selected_images) < 2:
        messagebox.showerror("Error", "Please select at least two images to stitch.")
        return

    images = [load_image(img_path) for img_path in selected_images]
    result_img = images[0]

    def update_result(image):
        nonlocal result_img
        result_bgr = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result_bgr.thumbnail((400, 200))
        display_img = ImageTk.PhotoImage(result_bgr)
        result_img = image
        result_label.config(image=display_img)
        result_label.image = display_img

    def display_option(stitched_image, display_zone):
        result_bgr = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
        result_bgr.thumbnail((400, 200))
        result_img = ImageTk.PhotoImage(result_bgr)

        # Display zone
        display_zone.config(image=result_img)
        display_zone.image = result_img

    for i in range(1, len(images)):
        src_img = result_img
        tar_img = images[i]

        src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)

        SIFT_detector = cv2.SIFT_create()
        kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
        kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        rawMatches = bf.knnMatch(des1, des2, k=2)

        matches = []
        ratio = 0.75
        for m, n in rawMatches:
            if m.distance < n.distance * ratio:
                matches.append(m)

        kp1 = np.float32([kp.pt for kp in kp1])
        kp2 = np.float32([kp.pt for kp in kp2])

        pts1 = np.float32([kp1[m.queryIdx] for m in matches])
        pts2 = np.float32([kp2[m.trainIdx] for m in matches])

        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        print(f"Homography Matrix for step {i}:\n{H}")

        h1, w1 = src_img.shape[:2]
        h2, w2 = tar_img.shape[:2]

        stitched_images = [
            cv2.warpPerspective(src_img, H, (w1 + w2, h1)),
            cv2.warpPerspective(tar_img, np.linalg.inv(H), (w2 + w1, h2)),
        ]

        for idx, stitched in enumerate(stitched_images):
            stitched[0:h2, 0:w2] = tar_img if idx == 0 else src_img

        first_stitched = stitched_images[0]
        second_stitched = stitched_images[1]
        display_option(first_stitched, first_option)
        display_option(second_stitched, second_option)

        btn1 = tk.Button(
            button_frame,
            text=f"Select Image {i}-1",
            command=lambda: update_result(first_stitched),
        )
        btn2 = tk.Button(
            button_frame,
            text=f"Select Image {i}-2",
            command=lambda: update_result(second_stitched),
        )
        btn1.pack(side=tk.LEFT, padx=10)
        btn2.pack(side=tk.LEFT, padx=10)

        def handle_save(image):
            file_name = simpledialog.askstring(
                "Input", "Enter the file name:", parent=save_frame
            )

            if file_name:
                cv2.imwrite(f"{file_name}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo(
                    "Success", f"Final stitching complete. Saved as {file_name}.jpg."
                )
            else:
                messagebox.showwarning(
                    "No Name", "No file name entered. Save operation cancelled."
                )

        save_frame = tk.Frame()

        save_btn = tk.Button(
            btn_frame,
            text="Save image",
            command=lambda: handle_save(result_img),
        )

        save_btn.pack()


def select_images():
    files = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")),
    )
    for file in files:
        if file not in selected_images:
            selected_images.append(file)
            display_selected(file)


def display_selected(file):
    img = Image.open(file)
    img.thumbnail((100, 100))
    img = ImageTk.PhotoImage(img)
    label = tk.Label(selected_frame, image=img)
    label.image = img
    label.pack(side=tk.LEFT, padx=5, pady=5)


def reset():
    global selected_images
    selected_images = []
    result_label.config(text="Stitched result will appear here.")

    # Destroy all widgets inside selected_frame
    for widget in selected_frame.winfo_children():
        widget.destroy()  # Remove all widgets in the selected_frame

    # Destroy all widgets inside option_frame
    for widget in option_frame.winfo_children():
        widget.destroy()  # Remove all widgets in the option_frame

    # Re-add default widgets to option_frame (optional)
    default_first_option = tk.Label(
        option_frame, text="Option 1", bg="gray", width=400, height=200
    )
    default_first_option.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

    default_second_option = tk.Label(
        option_frame, text="Option 2", bg="gray", width=400, height=200
    )
    default_second_option.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)


# Initialize Tkinter root
root = tk.Tk()
root.title("Image Stitching with SIFT")
root.geometry("1400x800")

# Selected images
selected_images = []

# Create UI components
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Select Images", command=select_images).pack(
    side=tk.LEFT, padx=10
)
tk.Button(btn_frame, text="Stitch Images", command=stitch_images).pack(
    side=tk.LEFT, padx=10
)
tk.Button(btn_frame, text="Clear", command=reset).pack(side=tk.LEFT, padx=10)

selected_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=2)
selected_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Frame chứa các tùy chọn (first_option và second_option)
option_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=2)
option_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Label cho First Option và Second Option
first_option = tk.Label(option_frame, text="Option 1", bg="gray", width=400, height=200)
first_option.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

second_option = tk.Label(
    option_frame, text="Option 2", bg="gray", width=400, height=200
)
second_option.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

# Frame chứa các nút chọn
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Label cho Result nằm ở dưới
result_label = tk.Label(
    root, text="Stitched result will appear here.", bg="gray", width=400, height=200
)
result_label.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)


# Start Tkinter loop
root.mainloop()
