import cv2
import numpy as np


def harris_corner_detection(reference_image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    reference_image[dst > threshold * dst.max()] = [0, 0, 255]

    return reference_image


#sift image alignment
def align_image(image_to_align, reference_image, max_features=10, good_match_percent=0.7):

    im1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    kp2, des2 = sift.detectAndCompute(im2_gray, None)


    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < good_match_percent * n.distance:
                good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]

    #extract location of good matches
    if len(good_matches) > 5:
        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        height, width, channels = reference_image.shape

        aligned_image = cv2.warpPerspective(image_to_align, homography, (width, height))

        #create image with matches drawn between keypoints
        stitched_image = cv2.drawMatches(
            image_to_align, kp1,
            reference_image, kp2,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("matches.jpg", stitched_image)

        return aligned_image

if __name__ == "__main__":

    reference_image = cv2.imread("reference_img.png")

    harris_image = harris_corner_detection(reference_image, block_size=2, ksize=3, k=0.04, threshold=0.01)
    cv2.imwrite("harris.png", harris_image)

    image_to_align = cv2.imread("align_this.jpg")
    reference_image = cv2.imread("reference_img.png")
    aligned_image = align_image(image_to_align, reference_image, max_features=10, good_match_percent=0.7)
    cv2.imwrite("aligned.jpg", aligned_image)