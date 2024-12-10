import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def roberts_operator(image):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])

    grad_x = convolve(image, roberts_x)
    grad_y = convolve(image, roberts_y)

    grad_magnitude = np.hypot(grad_x, grad_y)
    return grad_magnitude

def sobel_operator(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    grad_magnitude = np.hypot(grad_x, grad_y)
    return grad_magnitude

image_path = 'C:\\Users\\Administrator\\Downloads\\kelapa.jpg' 
image = imageio.imread(image_path)

if len(image.shape) == 3:
    image = np.mean(image, axis=2)

image = image / 255.0

edges_roberts = roberts_operator(image)

edges_sobel = sobel_operator(image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges_roberts, cmap='gray')
plt.title('Deteksi Tepi (Roberts)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges_sobel, cmap='gray')
plt.title('Deteksi Tepi (Sobel)')
plt.axis('off')

plt.show()
