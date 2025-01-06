import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import imageio
import imutils


def select_images():
    files = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")),
    )
    for file in files:
        if file not in selected_images:
            selected_images.append(file)
            display_selected(file)


def display_first_option(stitched_image):
    # Create the first option label
    first_option = tk.Label(
        option_frame, text="Option 1", bg="gray", width=400, height=200
    )
    first_option.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

    # Update result_label to display the final stitched image
    result_rgb = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    result_rgb.thumbnail((400, 200))
    
    display_img = ImageTk.PhotoImage(result_rgb)

    # Update the label with the image
    first_option.config(image=display_img)
    first_option.image = display_img


def display_second_option(stitched_image):
    second_option = tk.Label(
        option_frame, text="Option 2", bg="gray", width=400, height=200
    )
    second_option.pack(side=tk.RIGHT, padx=10, pady=10)

    # Update result_label to display the final stitched image
    result_bgr = Image.fromarray(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    result_bgr.thumbnail((400, 200))
    display_img = ImageTk.PhotoImage(result_bgr)

    # Update the label with the image
    second_option.config(image=display_img)
    second_option.image = display_img


def load_image(file_path):
    """Load image and convert it to RGB for displaying in Tkinter."""
    return imageio.imread(file_path)


def handle_save(image, btn):
    file_name = simpledialog.askstring(
        "Input", "Enter the file name:", parent=save_frame
    )

    if file_name:
        cv2.imwrite(f"{file_name}.jpg", image)
        messagebox.showinfo(
            "Success",
            f"Final stitching complete. Saved as {file_name}.jpg.",
        )
        btn.destroy()
    else:
        messagebox.showwarning(
            "No Name", "No file name entered. Save operation cancelled."
        )


def stitch_images():
    if len(selected_images) < 2:
        messagebox.showerror("Error", "Please select at least two images to stitch.")
        return
    elif len(selected_images) == 2:
        images = [load_image(img_path) for img_path in selected_images]
        result_img = images[0]

        def update_result(image):
            nonlocal result_img
            image = post_process(image)
            result_bgr = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            result_bgr.thumbnail((800, 400))
            display_img = ImageTk.PhotoImage(result_bgr)
            result_img = image
            result_label.config(image=display_img)
            result_label.image = display_img

        src_img = result_img
        tar_img = images[1]

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

        h1, w1 = src_img.shape[:2]
        h2, w2 = tar_img.shape[:2]

        # Create larger canvas to handle both vertical and horizontal stitching
        stitched_width = max(w1, w2) * 2  # Double the max width
        stitched_height = max(h1, h2) * 2

        # Warp both images to the canvas
        stitched_images = [
            cv2.warpPerspective(src_img, H, (stitched_width, stitched_height)),
            cv2.warpPerspective(
                tar_img, np.linalg.inv(H), (stitched_width, stitched_height)
            ),
        ]

        for idx, stitched in enumerate(stitched_images):
            if idx == 0:
                stitched[0:h2, 0:w2] = (
                    tar_img  # Place the second image onto the canvas
                )
            else:
                stitched[0:h1, 0:w1] = (
                    src_img  # Place the first image onto the canvas
                )

        first_stitched = stitched_images[0][100:-50, :]
        second_stitched = stitched_images[1][100:-50, :]

        display_first_option(cv2.cvtColor(first_stitched, cv2.COLOR_BGR2RGB))
        display_second_option(cv2.cvtColor(second_stitched, cv2.COLOR_BGR2RGB))

        btn1 = tk.Button(
            button_frame,
            text=f"Select Image 1",
            command=lambda: update_result(cv2.cvtColor(first_stitched, cv2.COLOR_BGR2RGB)),
        )
        btn2 = tk.Button(
            button_frame,
            text=f"Select Image 2",
            command=lambda: update_result(cv2.cvtColor(second_stitched, cv2.COLOR_BGR2RGB)),
        )
        btn1.pack(side=tk.LEFT, padx=10)
        btn2.pack(side=tk.LEFT, padx=10)

        save_btn = tk.Button(
            btn_frame,
            text="Save image",
            command=lambda: handle_save(
                result_img, save_btn
            ),
        )

        save_btn.pack()
    else:
        images = []

        for image in selected_images:
            img = cv2.imread(image)
            if img is None:
                print(f"Can not read image: {image}")
                continue
            images.append(img)

        image_stitcher = cv2.Stitcher_create()
        error, stitched_img = image_stitcher.stitch(images)

        if error == cv2.Stitcher_OK:
            stitched_img = cv2.copyMakeBorder(
                stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
            )
            gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

            thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            contours = cv2.findContours(
                thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contours = imutils.grab_contours(contours)
            area_oi = max(contours, key=cv2.contourArea)

            mask = np.zeros(thresh_img.shape, dtype="uint8")
            x, y, w, h = cv2.boundingRect(area_oi)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            min_rectangle = mask.copy()
            sub = mask.copy()

            while cv2.countNonZero(sub) > 0:
                min_rectangle = cv2.erode(min_rectangle, None)
                sub = cv2.subtract(min_rectangle, thresh_img)

            contours = cv2.findContours(
                min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contours = imutils.grab_contours(contours)
            area_oi = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(area_oi)

            stitched_img = stitched_img[y : y + h, x : x + w]

            # Update result_label to display the final stitched image
            result_bgr = Image.fromarray(cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB))
            final_image = cv2.cvtColor(np.array(result_bgr), cv2.COLOR_RGB2BGR)
            result_bgr.thumbnail((400, 200))  # Resize to fit the label
            display_img = ImageTk.PhotoImage(result_bgr)
            result_label.config(image=display_img)
            result_label.image = display_img
            save_btn = tk.Button(
                btn_frame,
                text="Save image",
                command=lambda: handle_save(final_image, save_btn),
            )

            save_btn.pack()
        else:
            print("Error in stitching images:", error)  # Label for result at the bottom


def replace_black_with_average(stitched_img, x, y, w, h):
    # Iterate over each pixel in the specified region
    for i in range(x, x + w):
        for j in range(y, y + h):
            # Check if the pixel is black (RGB = 0, 0, 0) or close to black
            if np.all(stitched_img[j, i] == 0):  # Verify if the pixel is black
                neighbors = []

                # Gather surrounding pixels (within a certain range)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        # Ensure the neighboring pixel is within image bounds
                        if (
                            0 <= ni < stitched_img.shape[1]
                            and 0 <= nj < stitched_img.shape[0]
                        ):
                            # Check if the neighboring pixel is not black
                            if not np.all(stitched_img[nj, ni] == 0):
                                neighbors.append(stitched_img[nj, ni])

                # If at least one non-black pixel exists, calculate the average color and replace
                if neighbors:
                    avg_color = np.mean(neighbors, axis=0).astype(
                        int
                    )  # Compute the average of the surrounding colors
                    stitched_img[j, i] = (
                        avg_color  # Replace the black pixel with the average color
                    )

    return stitched_img


def post_process(stitched_img):
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(
        thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)

    area = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")

    x, y, w, h = cv2.boundingRect(area)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    min_rectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 5000:
        min_rectangle = cv2.erode(min_rectangle, None)
        sub = cv2.subtract(min_rectangle, thresh_img)

    contours = cv2.findContours(
        min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    area_oi = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(area_oi)

    stitched_img = stitched_img[y : y + h, x : x + w]

    stitched_img = replace_black_with_average(stitched_img, 0, 0, w, h)

    return stitched_img


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

    for widget in selected_frame.winfo_children():
        widget.destroy()

    for widget in option_frame.winfo_children():
        widget.destroy()

    for widget in result_label.winfo_children():
        widget.destroy()

    result_label.config(image=None)
    result_label.image = None
    result_label.config(text="Result image here...")


# Initialize Tkinter root
root = tk.Tk()
root.title("Image Stitching")
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

option_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=6)
option_frame.pack(expand=True, padx=10, pady=10)


button_frame = tk.Frame(root)
button_frame.pack(pady=10)

result_label = tk.Label(
    root,
    text="Stitched result will appear here.",
    width=800,
    height=400,
)
result_label.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

save_frame = tk.Frame()
root.mainloop()
