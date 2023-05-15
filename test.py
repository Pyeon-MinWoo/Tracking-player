import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('C:/Users/pyeonmu/Desktop/test/data/111.mp4')

# Define color range for fireworks (in this example, red and yellow)
# 불꽃놀이의 색상 범위 정의 (이 예에서, 빨간색과 노란색)
lower_color = np.array([0, 50, 50])
upper_color = np.array([30, 255, 255])

# Define image processing parameters
blur_kernel_size = (3, 3)
edge_threshold1 = 50
edge_threshold2 = 150
canny_aperture_size = 3

# Loop over frames in video
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth image
    blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2, apertureSize=canny_aperture_size)

    # Dilate the edges to thicken the lines
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply color threshold to isolate fireworks
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    fireworks = cv2.bitwise_and(frame, frame, mask=mask)

    # Merge the fireworks and edges images
    merged = cv2.addWeighted(fireworks, 1, cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR), 1, 0)

    # Display original and processed frames
    cv2.imshow('Original', frame)
    cv2.imshow('Processed', merged)

    # Wait for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
