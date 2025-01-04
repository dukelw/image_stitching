import tkinter as tk
from tkinter import filedialog, messagebox
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

    # # Load images and convert to grayscale
    # images = [load_image(img_path) for img_path in selected_images]
    # gray_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    src_img = load_image(selected_images[1])
    tar_img = load_image(selected_images[0])

    # Convert images to grayscale
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)

    # Initialize SIFT detector
    SIFT_detector = cv2.SIFT_create()
    kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
    kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

    # Use BFMatcher with KNN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    rawMatches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    matches = []
    ratio = 0.75
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)

    # Sort matches by distance and keep the top 200
    matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    matches = matches[:200]

    # Draw matches
    img3 = cv2.drawMatches(
        src_img,
        kp1,
        tar_img,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Extract keypoints
    kp1 = np.float32([kp.pt for kp in kp1])
    kp2 = np.float32([kp.pt for kp in kp2])

    pts1 = np.float32([kp1[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2[m.trainIdx] for m in matches])

    # Estimate homography
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    print("Homography Matrix:")
    print(H)

    # Warp perspective
    h1, w1 = src_img.shape[:2]
    h2, w2 = tar_img.shape[:2]

    result = cv2.warpPerspective(src_img, H, (w1 + w2, h1))
    result[0:h2, 0:w2] = tar_img

    # Convert result from RGB to BGR before saving with OpenCV
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("panorama_image.png", result_bgr)

    result_pil = Image.fromarray(result_bgr)
    result_pil.thumbnail((800, 400))  # Resize to fit result label
    result_img = ImageTk.PhotoImage(result_pil)

    # Update result_label with stitched image
    result_label.config(image=result_img)
    result_label.image = result_img

    # Save stitched image
    stitched_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("stitched_result_sift.jpg", stitched_bgr)
    messagebox.showinfo(
        "Success", "Stitching complete. Saved as stitched_result_sift.jpg."
    )


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


# Initialize Tkinter root
root = tk.Tk()
root.title("Image Stitching with SIFT")
root.geometry("900x600")

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

selected_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=2)
selected_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

result_label = tk.Label(
    root, text="Stitched result will appear here.", bg="gray", width=800, height=400
)
result_label.pack(pady=10)

# Start Tkinter loop
root.mainloop()
