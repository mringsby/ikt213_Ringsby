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

# Create SIFT detector
sift = cv.SIFT_create(
    nfeatures=5000,        # Increase from default 0 (unlimited)
    nOctaveLayers=4,       # Default is 3, try 4-5 for more keypoints
    contrastThreshold=0.04, # Lower = more keypoints (default 0.04)
    edgeThreshold=10,      # Higher = fewer edge keypoints (default 10)
    sigma=1.6             # Gaussian blur sigma (default 1.6)
)

# Create FLANN parameters
FLANN_INDEX_KDTREE = 1  # Algorithm type for SIFT/SURF
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # K-D Tree with 5 trees
search_params = dict(checks=50)  # Number of times the tree is recursively traversed

# Initialize FLANN matcher
flann = cv.FlannBasedMatcher(index_params, search_params)

for idx, (img1_path, img2_path) in enumerate(get_image_pairs()):
    print(f"Processing pair {idx + 1}: {img1_path} vs {img2_path}")

    # Load images
    image1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    image2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print(f"Could not load {img1_path} or {img2_path}")
        continue

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        print(f"No descriptors found for {img1_path} or {img2_path}")
        continue

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"Found {len(good_matches)} good matches")

    # Draw matches between images (shows both images side by side with connecting lines)
    matched_image = cv.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:50], None, flags=2)

    # Save the result with cleaner naming
    if idx == 0:  # Working directory images
        out_name = f'working_dir_sift.jpg'
    else:
        folder_name = os.path.dirname(img1_path)
        out_name = f'{folder_name}_sift.jpg'

    cv.imwrite(out_name, matched_image)
    print(f"Saved result as: {out_name}")

print("SIFT matching completed!")

# Display and save result
#cv.imshow('SIFT Matches', matched_image)
cv.waitKey(0)
cv.destroyAllWindows()
