import numpy as np
import cv2
import imutils

# Chỉ định danh sách các ảnh
image_paths = [
    "images/scene1_a.jpg",
    "images/scene1_b.jpg",
    "images/scene1_c.jpg",
    # Thêm nhiều ảnh nếu cần
]

images = []

# Đọc và hiển thị các ảnh
for image in image_paths:
    img = cv2.imread(image)
    if img is None:
        print(f"Không thể đọc ảnh: {image}")
        continue
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

# Tạo stitcher và ghép ảnh
image_stitcher = cv2.Stitcher_create()
error, stitched_img = image_stitcher.stitch(images)

# Kiểm tra lỗi và lưu ảnh đã ghép
if error == cv2.Stitcher_OK:
    cv2.imwrite(
        "stitched_image.jpg", stitched_img
    )  # Thêm phần mở rộng như .jpg hoặc .png
    cv2.imshow("Stitched Image", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Thêm viền vào ảnh đã ghép
    stitched_img = cv2.copyMakeBorder(
        stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Threshold Image", thresh_img)
    cv2.waitKey(0)

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

    cv2.imshow("Min Rectangle", min_rectangle)
    cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(area_oi)

    # Cắt ảnh ghép theo hình chữ nhật nhỏ nhất
    stitched_img = stitched_img[y : y + h, x : x + w]

    cv2.imwrite("stitched_output_process.png", stitched_img)
    cv2.imshow("Stitched Output Process", stitched_img)
    cv2.waitKey(0)
else:
    print("Error in stitching images:", error)
