"""
Basic MRI reconstruction example using the 2D Fourier transform.

This script creates a simple synthetic image, transforms it into
k-space using a 2D FFT, and reconstructs the image using the
inverse 2D FFT.

Only synthetic data are used.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Create a simple synthetic image
# ------------------------------------------------------------

image_size = 128

image = np.zeros((image_size, image_size), dtype=float)

x, y = np.meshgrid(
    np.arange(image_size),
    np.arange(image_size),
)

# Large circular object
region_1 = (
    (x - 64) ** 2
    + (y - 64) ** 2
    <= 30 ** 2
)

# Smaller object with different signal intensity
region_2 = (
    (x - 82) ** 2
    + (y - 50) ** 2
    <= 10 ** 2
)

image[region_1] = 1.0
image[region_2] = 1.8


# ------------------------------------------------------------
# 2. Transform the image into k-space
# ------------------------------------------------------------

kspace = np.fft.fftshift(
    np.fft.fft2(
        np.fft.ifftshift(image)
    )
)


# ------------------------------------------------------------
# 3. Prepare k-space for visualization
# ------------------------------------------------------------

# k-space has a very large dynamic range.
# Log scaling makes both strong and weak spatial-frequency
# components visible.
kspace_log = np.log1p(
    np.abs(kspace)
)


# ------------------------------------------------------------
# 4. Reconstruct the image from k-space
# ------------------------------------------------------------

reconstructed_complex = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(kspace)
    )
)

reconstructed_image = np.abs(
    reconstructed_complex
)


# ------------------------------------------------------------
# 5. Calculate reconstruction error
# ------------------------------------------------------------

reconstruction_error = np.mean(
    np.abs(image - reconstructed_image)
)

print(
    f"Mean absolute reconstruction error: "
    f"{reconstruction_error:.6e}"
)


# ------------------------------------------------------------
# 6. Visualize results
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))
plt.imshow(image, cmap="gray")
plt.title("Synthetic Image")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(kspace_log, cmap="gray")
plt.title("K-space (Log Magnitude)")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")
plt.tight_layout()


plt.show()
