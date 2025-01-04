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


# Utility function to plot multiple images
def plot_imgs(imgs, cols=5, size=7, title=""):
    rows = len(imgs) // cols + 1
    fig = plt.figure(figsize=(cols * size, rows * size))
    for i, img in enumerate(imgs):
        cmap = "gray" if len(img.shape) == 2 else None
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


# Load images from URLs
# src_img = imageio.imread("http://www.ic.unicamp.br/~helio/imagens_registro/foto1A.jpg")
# tar_img = imageio.imread("http://www.ic.unicamp.br/~helio/imagens_registro/foto1B.jpg")
src_img = imageio.imread("images/c2.jpg")
tar_img = imageio.imread("images/c1.jpg")

# Convert images to grayscale
src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)

# Plot the source and target images
plot_imgs([src_img, tar_img], size=8)

# Initialize SIFT detector
SIFT_detector = cv2.SIFT_create()
kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

# Plot keypoints
plot_imgs(
    [cv2.drawKeypoints(src_img, kp1, None), cv2.drawKeypoints(tar_img, kp2, None)]
)

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
plot_img(img3, size=(15, 10))

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

# Plot the final result
plot_img(result, size=(20, 10))
# Convert result from RGB to BGR before saving with OpenCV
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("stiched_image.png", result_bgr)