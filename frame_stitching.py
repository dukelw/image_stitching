import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils

video_frames = []
video_path = ""
video_duration = 0

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

# Function to select a video and extract details
def select_video():
    global video_path, video_duration

    video_path = filedialog.askopenfilename(
        title="Select Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if not video_path:
        return

    # Capture video to get details
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the video file.")
        return

    # Get video duration
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames // frame_rate

    cap.release()

    # Update video duration and preview
    video_duration_label.config(text=f"Video Duration: {video_duration} seconds")
    display_video_preview(video_path)

# Function to display video preview
def display_video_preview(path):
    cap = cv2.VideoCapture(path)
    success, frame = cap.read()
    cap.release()

    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((400, 300))
        img_tk = ImageTk.PhotoImage(image)

        preview_label.config(width=200, height=100)
        preview_label.config(image=img_tk)
        preview_label.image = img_tk
    else:
        messagebox.showerror("Error", "Failed to load video preview.")

# Function to extract frames based on user input
def extract_frames():
    global video_frames

    if not video_path:
        messagebox.showerror("Error", "No video selected.")
        return

    try:
        interval = int(interval_input.get())
        if interval <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid interval (positive integer).")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the video file.")
        return

    video_frames = []
    for second in range(0, video_duration, interval):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            cropped_frame = frame_rgb[:, 200:width - 200]
            video_frames.append(cropped_frame)

    cap.release()

    if video_frames:
        display_frames()
    else:
        messagebox.showerror("Error", "No frames extracted.")

# Function to display the extracted frames
def display_frames():
    for widget in frame_display_area.winfo_children():
        widget.destroy()

    for i, frame in enumerate(video_frames):
        image = Image.fromarray(frame)
        image.thumbnail((200, 150))
        img_tk = ImageTk.PhotoImage(image)

        img_label = tk.Label(frame_display_area, image=img_tk)
        img_label.image = img_tk
        img_label.grid(row=i // 5, column=i % 5, padx=5, pady=5)

# Stitching logic remains the same
# Function to stitch frames
def stitch_video():
    image_stitcher = cv2.Stitcher_create()
    error, stitched_img = image_stitcher.stitch(video_frames)
    cv2.imwrite("original.png", stitched_img)

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

# Reset the UI
def reset():
    global video_frames, video_path, video_duration
    video_frames = []
    video_path = ""
    video_duration = 0

    for widget in frame_display_area.winfo_children():
        widget.destroy()

    preview_label.config(image=None)
    preview_label.image = None
    video_duration_label.config(text="Video Duration: N/A")
    result_label.config(image=None)
    result_label.image = None

# Initialize Tkinter root
root = tk.Tk()
root.title("Video Frame Extractor and Stitcher")
root.geometry("1200x800")

# Video preview and details
details_frame = tk.Frame(root)
details_frame.pack(pady=10)

preview_label = tk.Label(details_frame, text="Video Preview", width=20, height=10, relief=tk.SUNKEN, bd=2)
preview_label.pack(side=tk.LEFT, padx=10)

video_duration_label = tk.Label(details_frame, text="Video Duration: N/A")
video_duration_label.pack(side=tk.LEFT, padx=10)

# Interval input
tk.Label(details_frame, text="Frame Extraction Interval (seconds):").pack(side=tk.LEFT, padx=10)
interval_input = tk.Entry(details_frame, width=5)
interval_input.pack(side=tk.LEFT, padx=10)
interval_input.insert(0, "3")

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

select_btn = tk.Button(btn_frame, text="Select Video", command=select_video)
select_btn.pack(side=tk.LEFT, padx=5)

extract_btn = tk.Button(btn_frame, text="Extract Frames", command=extract_frames)
extract_btn.pack(side=tk.LEFT, padx=5)

stitch_btn = tk.Button(btn_frame, text="Stitch Video", command=stitch_video)
stitch_btn.pack(side=tk.LEFT, padx=5)

clear_btn = tk.Button(btn_frame, text="Clear", command=reset)
clear_btn.pack(side=tk.LEFT, padx=5)

save_frame = tk.Frame()

canvas_frame = tk.Frame(root)
canvas_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame)
scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
frame_display_area = tk.Frame(canvas)

frame_display_area.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=frame_display_area, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

result_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
result_frame.pack(pady=10, fill=tk.X)

result_label = tk.Label(result_frame, text="Stitched result will appear here.", bd=2, width=400, height=200)
result_label.pack(pady=10)

root.mainloop()
