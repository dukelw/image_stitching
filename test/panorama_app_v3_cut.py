import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import imageio
import imutils
from image_crop_tool import ImageCropTool
import math


def select_images():
    files = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*")),
    )
    for file in files:
        if file not in selected_images:
            selected_images.append(file)
            display_selected(file)


# def load_image(file_path):
#     """Load image and convert it to RGB for displaying in Tkinter."""
#     return imageio.imread(file_path)


import imageio
from PIL import Image


import imageio
import cv2
import numpy as np


import cv2
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
        image = post_process(image)
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

        # Create larger canvas to handle both vertical and horizontal stitching
        stitched_width = max(w1, w2) * 2  # Double the max width
        stitched_height = max(h1, h2)

        # Warp both images to the canvas
        stitched_images = [
            cv2.warpPerspective(src_img, H, (stitched_width, stitched_height)),
            cv2.warpPerspective(
                tar_img, np.linalg.inv(H), (stitched_width, stitched_height)
            ),
        ]

        for idx, stitched in enumerate(stitched_images):
            if idx == 0:
                stitched[0:h2, 0:w2] = tar_img  # Place the second image onto the canvas
            else:
                stitched[0:h1, 0:w1] = src_img  # Place the first image onto the canvas

        first_stitched = stitched_images[0][100:-50, :]
        second_stitched = stitched_images[1][100:-50, :]

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


# def post_process(stitched_img):
#     # Chuyển ảnh sang grayscale
#     gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

#     # Threshold hóa ảnh
#     thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

#     # Tìm các contours
#     contours = cv2.findContours(
#         thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     contours = imutils.grab_contours(contours)

#     # Chọn contour có diện tích lớn nhất
#     area_oi = max(contours, key=cv2.contourArea)

#     # Tạo một mask trắng với kích thước ảnh ban đầu
#     mask = np.zeros(thresh_img.shape, dtype="uint8")

#     # Vẽ rectangle vào mask tại vị trí contour lớn nhất
#     x, y, w, h = cv2.boundingRect(area_oi)
#     cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

#     # Tạo min_rectangle bằng cách dãn nở mask
#     min_rectangle = mask.copy()
#     sub = mask.copy()

#     # Lặp lại việc xóa dần vùng trắng trong mask cho tới khi không còn gì
#     while cv2.countNonZero(sub) > 5000:
#         min_rectangle = cv2.erode(min_rectangle, None)
#         sub = cv2.subtract(min_rectangle, thresh_img)

#     # Lấy lại contour từ min_rectangle
#     contours = cv2.findContours(
#         min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#     contours = imutils.grab_contours(contours)
#     area_oi = max(contours, key=cv2.contourArea)

#     # Lấy tọa độ của vùng bao quanh ảnh không có dư màu đen
#     x, y, w, h = cv2.boundingRect(area_oi)

#     # Cắt ảnh theo bounding box
#     stitched_img = stitched_img[y : y + h, x : x + w]

#     return stitched_img


def replace_black_with_average(stitched_img, x, y, w, h):
    # Duyệt qua từng pixel trong vùng được chỉ định
    for i in range(x, x + w):
        for j in range(y, y + h):
            # Kiểm tra nếu pixel có màu đen (RGB = 0, 0, 0) hoặc gần đen
            if np.all(stitched_img[j, i] == 0):  # Kiểm tra nếu pixel có màu đen
                neighbors = []

                # Lấy các pixel xung quanh (trong phạm vi nhất định)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < stitched_img.shape[1]
                            and 0 <= nj < stitched_img.shape[0]
                        ):
                            if not np.all(
                                stitched_img[nj, ni] == 0
                            ):  # Kiểm tra nếu pixel không phải đen
                                neighbors.append(stitched_img[nj, ni])

                # Nếu có ít nhất một pixel không phải đen, tính trung bình và thay thế
                if neighbors:
                    avg_color = np.mean(neighbors, axis=0).astype(
                        int
                    )  # Tính trung bình của các màu
                    stitched_img[j, i] = (
                        avg_color  # Thay thế pixel đen bằng màu trung bình
                    )

    return stitched_img


def post_process(stitched_img):
    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    # Threshold hóa ảnh
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Tìm các contours
    contours = cv2.findContours(
        thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)

    # Chọn contour có diện tích lớn nhất
    area_oi = max(contours, key=cv2.contourArea)

    # Tạo một mask trắng với kích thước ảnh ban đầu
    mask = np.zeros(thresh_img.shape, dtype="uint8")

    # Vẽ rectangle vào mask tại vị trí contour lớn nhất
    x, y, w, h = cv2.boundingRect(area_oi)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Tạo min_rectangle bằng cách dãn nở mask
    min_rectangle = mask.copy()
    sub = mask.copy()

    # Lặp lại việc xóa dần vùng trắng trong mask cho tới khi không còn gì
    while cv2.countNonZero(sub) > 5000:
        min_rectangle = cv2.erode(min_rectangle, None)
        sub = cv2.subtract(min_rectangle, thresh_img)

    # Lấy lại contour từ min_rectangle
    contours = cv2.findContours(
        min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    area_oi = max(contours, key=cv2.contourArea)

    # Lấy tọa độ của vùng bao quanh ảnh không có dư màu đen
    x, y, w, h = cv2.boundingRect(area_oi)

    # Cắt ảnh theo bounding box
    stitched_img = stitched_img[y : y + h, x : x + w]

    # # Bây giờ ta cần kiểm tra các vùng pixel có màu đen và loại bỏ chúng (chỉ cắt những vùng chứa màu đen)
    # for i in range(x, x + w):
    #     for j in range(y, y + h):
    #         # Nếu pixel có màu đen (hoặc gần đen), chúng ta sẽ loại bỏ
    #         if np.all(
    #             stitched_img[j, i] == 0
    #         ):  # Kiểm tra nếu pixel có màu đen (RGB = 0,0,0)
    #             stitched_img[j, i] = [255, 255, 255]  # Chuyển pixel đó thành màu trắng

    # return stitched_img

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
