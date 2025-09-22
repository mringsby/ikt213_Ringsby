import cv2 as cv
import numpy as np
import os

def get_image_pairs():
    pairs = []
    # Working directory images - exclude our output files
    images = sorted([f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
                    and not ('_orb.jpg' in f or '_sift.jpg' in f)])
    if len(images) >= 2:
        pairs.append((images[0], images[1]))

    # Folders: different_1..10 and same_1..10
    for prefix in ['different_', 'same_']:
        for i in range(1, 11):
            folder = f"{prefix}{i}"
            if os.path.isdir(folder):
                imgs = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                if len(imgs) >= 2:
                    pairs.append((os.path.join(folder, imgs[0]), os.path.join(folder, imgs[1])))
    return pairs

# Create ORB detector with additional parameters
orb = cv.ORB_create(
    nfeatures=5000,        # Maximum number of features to retain
    scaleFactor=1.2,       # Pyramid decimation ratio (default 1.2)
    nlevels=8,             # Number of pyramid levels (default 8)
    edgeThreshold=31,      # Size of border where features not detected (default 31)
    firstLevel=0,          # Level of pyramid to put source image (default 0)
    WTA_K=2,               # Number of points for oriented BRIEF (default 2)
    scoreType=cv.ORB_HARRIS_SCORE,  # HARRIS_SCORE or FAST_SCORE
    patchSize=31,          # Size of patch used by oriented BRIEF (default 31)
    fastThreshold=20       # FAST threshold for corner detection (default 20)
)

for idx, (img1_path, img2_path) in enumerate(get_image_pairs()):
    print(f"Processing pair {idx + 1}: {img1_path} vs {img2_path}")

    # Load images
    image1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print(f"Could not load {img1_path} or {img2_path}")
        continue

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        print(f"No descriptors found for {img1_path} or {img2_path}")
        continue

    #BF matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < 38]
    print(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")

    matched_image = cv.drawMatches(image1, keypoints1, image2, keypoints2,good_matches[:100], None, flags=2)

    # Save the result with cleaner naming
    if idx == 0:  # Working directory images
        out_name = f'working_dir_orb.jpg'
    else:
        folder_name = os.path.dirname(img1_path)
        out_name = f'{folder_name}_orb.jpg'

    cv.imwrite(out_name, matched_image)
    print(f"Saved result as: {out_name}")

print("ORB matching completed!")
