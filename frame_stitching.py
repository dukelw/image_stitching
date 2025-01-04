import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils

video_frames = []


# Function to select a video and extract frames
def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if not video_path:
        return

    # Extract frames from the video
    frames = extract_frames(video_path)
    if not frames:
        messagebox.showerror("Error", "Failed to extract frames from the video.")
        return

    global video_frames
    video_frames = frames

    # Display the frames
    display_frames()


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // frame_rate

    frames = []
    for second in range(0, duration, 3):  # Extract frames every 3 seconds
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)  # Set to each 3rd second
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Crop 200px from the left and right
            height, width = frame_rgb.shape[:2]
            cropped_frame = frame_rgb[
                :, 200 : width - 200
            ]  # Keep the center part of the frame

            frames.append(cropped_frame)

    cap.release()
    print(f"Total frames: {len(frames)}")
    return frames


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


# Function to display the extracted frames
def display_frames():
    if not video_frames:
        messagebox.showerror("Error", "No frames to display.")
        return

    # Clear the frame display area
    for widget in frame_display_area.winfo_children():
        widget.destroy()

    for i, frame in enumerate(video_frames):
        # Convert frame to ImageTk format
        image = Image.fromarray(frame)
        image.thumbnail((200, 150))  # Resize for display
        img_tk = ImageTk.PhotoImage(image)

        # Create a label to display the image
        img_label = tk.Label(frame_display_area, image=img_tk)
        img_label.image = img_tk  # Keep reference to avoid garbage collection
        img_label.grid(row=i // 4, column=i % 4, padx=5, pady=5)


def stitch_video():
    images = []

    for image in video_frames:
        images.append(image)

    image_stitcher = cv2.Stitcher_create()
    print(len(images))
    error, stitched_img = image_stitcher.stitch(images)
    cv2.imwrite("hello.png", stitched_img)

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

        # The image remains in BGR format, no need for conversion
        # Update result_label to display the final stitched image
        result_rgb = Image.fromarray(
            stitched_img
        )  # No need to convert, as it's still BGR
        result_rgb.thumbnail((400, 200))  # Resize to fit the label
        display_img = ImageTk.PhotoImage(result_rgb)
        result_label.config(image=display_img)
        result_label.image = display_img

        result_label.config(image=display_img)
        result_label.image = display_img

        # stitched_image is MatLike type bga (convert to rgb to display the right color)
        save_btn = tk.Button(
            btn_frame,
            text="Save image",
            command=lambda: handle_save(
                cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB), save_btn
            ),
        )

        save_btn.pack()
    else:
        print("Error in stitching frame:", error)


def reset():
    global video_frames
    video_frames = []

    for widget in frame_display_area.winfo_children():
        widget.destroy()

    result_label.config(image=None)
    result_label.image = None
    result_label.config(text="Result image here...")


# Initialize Tkinter root
root = tk.Tk()
root.title("Image Stitching with SIFT")
root.geometry("1400x800")

# Create UI components
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Select Video", command=select_video).pack(
    side=tk.LEFT, padx=10
)

tk.Button(btn_frame, text="Stitch Video", command=stitch_video).pack(
    side=tk.LEFT, padx=10
)
tk.Button(btn_frame, text="Clear", command=reset).pack(side=tk.LEFT, padx=10)


# Create a scrollable frame to display frames
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
frame_display_area = tk.Frame(canvas)

frame_display_area.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=frame_display_area, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Result label for the stitched image
result_label = tk.Label(
    root,
    text="Stitched result will appear here.",
    width=800,
    height=400,
    relief=tk.SUNKEN,
    bd=2,
)
result_label.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

save_frame = tk.Frame()
root.mainloop()
