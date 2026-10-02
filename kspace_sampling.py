"""
Demonstration of how different regions of k-space contribute
to MRI image reconstruction.

This script creates a synthetic image, transforms it into k-space,
and compares reconstruction using:

1. Full k-space
2. Central k-space only
3. Peripheral k-space only

Only synthetic data are used.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Create a synthetic image
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

# Smaller object with higher signal intensity
region_2 = (
    (x - 82) ** 2
    + (y - 50) ** 2
    <= 10 ** 2
)

image[region_1] = 1.0
image[region_2] = 1.8


# ------------------------------------------------------------
# 2. Transform image into centered k-space
# ------------------------------------------------------------

kspace = np.fft.fftshift(
    np.fft.fft2(
        np.fft.ifftshift(image)
    )
)


# ------------------------------------------------------------
# 3. Create a central k-space mask
# ------------------------------------------------------------

center_size = 32

center = image_size // 2
half_width = center_size // 2

central_mask = np.zeros_like(image)

central_mask[
    center - half_width:center + half_width,
    center - half_width:center + half_width
] = 1


# ------------------------------------------------------------
# 4. Keep only central k-space
# ------------------------------------------------------------

central_kspace = kspace * central_mask


# ------------------------------------------------------------
# 5. Keep only peripheral k-space
# ------------------------------------------------------------

peripheral_mask = 1 - central_mask

peripheral_kspace = kspace * peripheral_mask


# ------------------------------------------------------------
# 6. Reconstruction function
# ------------------------------------------------------------

def reconstruct(kspace_data):
    """Reconstruct magnitude image from centered k-space."""

    reconstructed_complex = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace_data)
        )
    )

    return np.abs(reconstructed_complex)


# ------------------------------------------------------------
# 7. Reconstruct images
# ------------------------------------------------------------

full_reconstruction = reconstruct(kspace)

central_reconstruction = reconstruct(
    central_kspace
)

peripheral_reconstruction = reconstruct(
    peripheral_kspace
)


# ------------------------------------------------------------
# 8. Display k-space masks
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))
plt.imshow(central_mask, cmap="gray")
plt.title("Central K-space Mask")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(peripheral_mask, cmap="gray")
plt.title("Peripheral K-space Mask")
plt.axis("off")
plt.tight_layout()


# ------------------------------------------------------------
# 9. Display reconstructed images
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))
plt.imshow(full_reconstruction, cmap="gray")
plt.title("Full K-space Reconstruction")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(central_reconstruction, cmap="gray")
plt.title("Central K-space Only")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(peripheral_reconstruction, cmap="gray")
plt.title("Peripheral K-space Only")
plt.axis("off")
plt.tight_layout()


plt.show()
