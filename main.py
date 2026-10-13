import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "image/landscape.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Check the image path:", image_path)
    exit()

print("Image loaded successfully!")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

height, width, channels = image.shape

print("\nImage Properties")
print("----------------")
print("Width:", width)
print("Height:", height)
print("Channels:", channels)
print("Resolution:", width, "x", height)
print("Data Type:", image.dtype)

cv2.imwrite("image/output.jpg", image)
cv2.imwrite("image/output.png", image)

print("\nImages saved successfully!")
print("JPEG: image/output.jpg")
print("PNG: image/output.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6, 4))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(hsv_display)
plt.title("HSV Image")
plt.axis("off")
plt.show()

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

lab_display = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(lab_display)
plt.title("LAB Image")
plt.axis("off")
plt.show()

resized = cv2.resize(image, (400, 300))

resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(resized_rgb)
plt.title("Resized Image")
plt.axis("off")
plt.show()

height, width = image.shape[:2]

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    45,
    1.0
)

rotated = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(rotated_rgb)
plt.title("Rotated Image - 45 Degrees")
plt.axis("off")
plt.show()

horizontal_flip = cv2.flip(image, 1)

horizontal_flip_rgb = cv2.cvtColor(
    horizontal_flip,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(6, 4))
plt.imshow(horizontal_flip_rgb)
plt.title("Horizontal Flip")
plt.axis("off")
plt.show()

vertical_flip = cv2.flip(image, 0)

vertical_flip_rgb = cv2.cvtColor(
    vertical_flip,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(6, 4))
plt.imshow(vertical_flip_rgb)
plt.title("Vertical Flip")
plt.axis("off")
plt.show()

negative = 255 - image

negative_rgb = cv2.cvtColor(
    negative,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(6, 4))
plt.imshow(negative_rgb)
plt.title("Negative Image")
plt.axis("off")
plt.show()

height, width = image.shape[:2]

x1 = width // 4
y1 = height // 4
x2 = 3 * width // 4
y2 = 3 * height // 4

roi = image[y1:y2, x1:x2]

roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(roi_rgb)
plt.title("Region of Interest (ROI)")
plt.axis("off")
plt.show()

roi_height, roi_width = roi.shape[:2]

print("\nROI Properties")
print("--------------")
print("ROI Width:", roi_width)
print("ROI Height:", roi_height)
print("ROI Resolution:", roi_width, "x", roi_height)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(resized_rgb)
plt.title("Resized")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(rotated_rgb)
plt.title("Rotated")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(horizontal_flip_rgb)
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(negative_rgb)
plt.title("Negative")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n================================")
print("Experiment 1 Completed!")
print("================================")