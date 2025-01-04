import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import imageio


# Utility function to plot an image
def plot_img(img, size=(7, 7), title=""):
    cmap = "gray" if len(img.shape) == 2 else None
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


# Load images from URLs
src_img = imageio.imread("http://www.ic.unicamp.br/~helio/imagens_registro/foto1A.jpg")
tar_img = imageio.imread("http://www.ic.unicamp.br/~helio/imagens_registro/foto1B.jpg")
# src_img = imageio.imread("images/image1.jpg")
# tar_img = imageio.imread("images/image2.jpg")

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

# Extract keypoints
kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

pts1 = np.float32([kp1[m.queryIdx] for m in matches])
pts2 = np.float32([kp2[m.trainIdx] for m in matches])

# Estimate homography
H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC)


# Warp perspective
h1, w1 = src_img.shape[:2]
h2, w2 = tar_img.shape[:2]

# Ensure the result height matches the target image height
result = cv2.warpPerspective(src_img, H, (w1 + w2, max(h1, h2)))

# Create a mask for blending
alpha = np.zeros((result.shape[0], result.shape[1]), dtype=np.float32)
alpha[0:h2, 0:w2] = np.tile(np.linspace(1, 0, w2), (h2, 1))

# Adjust target image to fit into the result
tar_img_resized = cv2.resize(tar_img, (result.shape[1], result.shape[0]))

# Perform blending
alpha_blended = np.dstack([alpha, alpha, alpha])  # Ensure 3 channels

result_blended = (
    result.astype(np.float32) * (1 - alpha_blended)
    + tar_img_resized.astype(np.float32) * alpha_blended
)
result_blended = np.clip(result_blended, 0, 255).astype(np.uint8)

# Add border
result_with_border = cv2.copyMakeBorder(
    result_blended, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
)
cv2.imshow("result_blended", result_with_border)
cv2.waitKey(0)

# Convert to grayscale and threshold
gray = cv2.cvtColor(result_with_border, cv2.COLOR_RGB2GRAY)
thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# Find contours
contours = cv2.findContours(
    thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(contours)
area_oi = max(contours, key=cv2.contourArea)

# Create mask
mask = np.zeros(thresh_img.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(area_oi)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# Refine rectangle with erosion
min_rectangle = mask.copy()
sub = mask.copy()
while cv2.countNonZero(sub) > 0:
    min_rectangle = cv2.erode(min_rectangle, None)
    sub = cv2.subtract(min_rectangle, thresh_img)

# Crop the final stitched image
contours = cv2.findContours(
    min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(contours)
area_oi = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(area_oi)
final_result = result_with_border[y : y + h, x : x + w]

# Save and display the final output
final_result_bgr = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
cv2.imwrite("v4_process_with_blending.png", final_result_bgr)
plot_img(final_result, size=(20, 10), title="Final Blended Stitched Image")
