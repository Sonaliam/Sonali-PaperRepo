import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur_image(input_image_path, k=15, lambda_val=0.25):
    # Read the input image
    img = cv2.imread(input_image_path)

    # Apply Gaussian blur with kernel size k and standard deviation lambda_val
    blurred_img = cv2.GaussianBlur(img, (k, k), lambda_val)

    # Show original and blurred images side by side
    plt.figure(figsize=(10, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Display blurred image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blurred Image")
    plt.axis("off")

    plt.show()


