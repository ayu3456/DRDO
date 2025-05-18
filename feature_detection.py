import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

def detect_features(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    return pts1, pts2, img_matches

if __name__ == "__main__":
    # Download sample images from the internet
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg", "image1.jpg")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right01.jpg", "image2.jpg")
    
    # Load images
    img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        exit()
    
    # Detect features and match
    pts1, pts2, img_matches = detect_features(img1, img2)
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches')
    plt.axis('off')
    # Save the figure to the results folder
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/feature_matches.png')
    plt.close()
    
    # Save the number of matches to a text file
    with open('results/feature_matches.txt', 'w') as f:
        f.write(f"Number of matches found: {len(pts1)}\n")
    
    print(f"Number of matches found: {len(pts1)} (saved to results/feature_matches.png)") 