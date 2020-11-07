import numpy as np
import cv2

mnist_image1 = np.load("D:\SciSharp\SharpCV\data\mnist_first_image_data.npy")
mnist_image1 = mnist_image1.reshape([28, 28]);
cv2.imshow("NDShow", mnist_image1);
cv2.waitKey(0)
