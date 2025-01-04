import cv2
import numpy as np


def initialize_mosaic(
    first_frame, output_height_times=3, output_width_times=1.2, detector_type="sift"
):
    detector, bf = initialize_detector(detector_type)

    output_img = np.zeros(
        shape=(
            int(output_height_times * first_frame.shape[0]),
            int(output_width_times * first_frame.shape[1]),
            first_frame.shape[2],
        )
    )

    w_offset = int(output_img.shape[0] / 1 - first_frame.shape[0] / 1)
    h_offset = int(output_img.shape[1] / 2 - first_frame.shape[1] / 2)

    output_img[
        w_offset : w_offset + first_frame.shape[0],
        h_offset : h_offset + first_frame.shape[1],
        :,
    ] = first_frame

    H_old = np.eye(3)
    H_old[0, 2] = h_offset
    H_old[1, 2] = w_offset

    return detector, bf, output_img, H_old, w_offset, h_offset


def initialize_detector(detector_type):
    if detector_type == "sift":
        detector = cv2.SIFT_create(700)
        bf = cv2.BFMatcher()
    elif detector_type == "orb":
        detector = cv2.ORB_create(700)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError("Unsupported detector type")

    return detector, bf


def detect_and_compute(detector, frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(frame_gray, None)
    return keypoints, descriptors


def match_descriptors(detector_type, bf, des_cur, des_prev):
    if detector_type == "sift":
        pair_matches = bf.knnMatch(des_cur, des_prev, k=2)
        matches = [m for m, n in pair_matches if m.distance < 0.7 * n.distance]
    elif detector_type == "orb":
        matches = bf.match(des_cur, des_prev)

    return sorted(matches, key=lambda x: x.distance)[:20]


def find_homography(image_1_kp, image_2_kp, matches):
    """gets two matches and calculate the homography between two images

    Args:
        image_1_kp (np array): keypoints of image 1
        image_2_kp (np_array): keypoints of image 2
        matches (np array): matches between keypoints in image 1 and image 2

    Returns:
        np arrat of shape [3,3]: Homography matrix
    """
    # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    for i in range(0, len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

    homography, mask = cv2.findHomography(
        image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0
    )

    return homography


def warp_frame(output_img, frame_cur, H):
    """warps the current frame based of calculated homography H

    Args:
        frame_cur (np array): current frame
        H (np array of shape [3,3]): homography matrix

    Returns:
        np array: image output of mosaicing
    """
    warped_img = cv2.warpPerspective(
        frame_cur,
        H,
        (output_img.shape[1], output_img.shape[0]),
        flags=cv2.INTER_LINEAR,
    )

    transformed_corners = get_transformed_corners(frame_cur, H)
    warped_img = draw_border(warped_img, transformed_corners)

    output_img[warped_img > 0] = warped_img[warped_img > 0]
    output_temp = np.copy(output_img)
    output_temp = draw_border(output_temp, transformed_corners, color=(0, 0, 255))

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", output_temp / 255.0)

    return output_img


def get_transformed_corners(frame_cur, H):
    """finds the corner of the current frame after warp

    Args:
        frame_cur (np array): current frame
        H (np array of shape [3,3]): Homography matrix

    Returns:
        [np array]: a list of 4 corner points after warping
    """
    corner_0 = np.array([0, 0])
    corner_1 = np.array([frame_cur.shape[1], 0])
    corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
    corner_3 = np.array([0, frame_cur.shape[0]])

    corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    transformed_corners = np.array(transformed_corners, dtype=np.int32)

    return transformed_corners


def draw_border(image, corners, color=(0, 0, 0)):
    """This functions draw rectancle border

    Args:
        image ([type]): current mosaiced output
        corners (np array): list of corner points
        color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).

    Returns:
        np array: the output image with border
    """
    for i in range(corners.shape[1] - 1, -1, -1):
        cv2.line(
            image,
            tuple(corners[0, i, :]),
            tuple(corners[0, i - 1, :]),
            thickness=5,
            color=color,
        )
    return image


def main():
    video_path = "videos/plane.mp4"
    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    detector, bf, output_img, H_old, w_offset, h_offset = initialize_mosaic(first_frame)

    kp_prev, des_prev = detect_and_compute(detector, first_frame)

    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            break

        kp_cur, des_cur = detect_and_compute(detector, frame_cur)
        matches = match_descriptors("sift", bf, des_cur, des_prev)

        if len(matches) >= 4:
            H = find_homography(kp_cur, kp_prev, matches)
            H = np.matmul(H_old, H)

            output_img = warp_frame(output_img, frame_cur, H)

            H_old = H
            kp_prev = kp_cur
            des_prev = des_cur

        small_frame = cv2.resize(frame_cur, (320, 240))
        cv2.imshow("Video", small_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite("mosaic.jpg", output_img)


if __name__ == "__main__":
    main()
