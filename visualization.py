import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from feature_detection import detect_features
from camera_pose import estimate_camera_pose
from triangulation import triangulate_points
import cv2
import os

def visualize_scene(img1, img2, K):
    # Detect features and get matched points
    pts1, pts2, img_matches = detect_features(img1, img2)
    
    # Estimate camera pose
    R, t, mask = estimate_camera_pose(img1, img2, K)
    
    # Triangulate points
    points_3d = triangulate_points(pts1, pts2, K, R, t)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot feature matches
    ax1 = fig.add_subplot(121)
    ax1.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    ax1.set_title('Feature Matches')
    ax1.axis('off')
    
    # Plot 3D points and camera poses
    ax2 = fig.add_subplot(122, projection='3d')
    # Only plot valid (finite) points
    valid = np.isfinite(points_3d).all(axis=1)
    ax2.scatter(points_3d[valid, 0], points_3d[valid, 1], points_3d[valid, 2], c='b', marker='.')
    
    # Plot camera poses
    # First camera at origin
    ax2.scatter([0], [0], [0], c='r', marker='^', s=100)
    
    # Second camera
    camera2_pos = -R.T @ t
    ax2.scatter([camera2_pos[0]], [camera2_pos[1]], [camera2_pos[2]], c='g', marker='^', s=100)
    
    # Set labels and title
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Reconstruction')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/visualization.png')
    plt.close()
    print('Visualization saved to results/visualization.png')

if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        exit()
    
    # Camera matrix (example values - should be calibrated for real use)
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    # Visualize the scene
    visualize_scene(img1, img2, K) 