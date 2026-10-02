"""
Demonstration of Cartesian k-space undersampling in MRI.

This script creates a synthetic image, transforms it into k-space,
and applies regular undersampling along one k-space direction.

Reconstructions are compared for:

1. Fully sampled k-space
2. Acceleration factor R = 2
3. Acceleration factor R = 4

Regular undersampling produces aliasing in the reconstructed image.

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

# Main circular object
region_1 = (
    (x - 64) ** 2
    + (y - 64) ** 2
    <= 30 ** 2
)

# Smaller high-intensity object
region_2 = (
    (x - 82) ** 2
    + (y - 50) ** 2
    <= 10 ** 2
)

# Additional off-center object
# This makes aliasing easier to visualize.
region_3 = (
    (x - 25) ** 2
    + (y - 90) ** 2
    <= 8 ** 2
)

image[region_1] = 1.0
image[region_2] = 1.8
image[region_3] = 1.4


# ------------------------------------------------------------
# 2. Transform image into centered k-space
# ------------------------------------------------------------

kspace = np.fft.fftshift(
    np.fft.fft2(
        np.fft.ifftshift(image)
    )
)


# ------------------------------------------------------------
# 3. Create Cartesian undersampling masks
# ------------------------------------------------------------

def create_undersampling_mask(size, acceleration):
    """
    Create a regular Cartesian undersampling mask.

    Every R-th k-space row is retained, where R is the
    acceleration factor.
    """

    mask = np.zeros((size, size), dtype=float)

    mask[::acceleration, :] = 1

    return mask


mask_r2 = create_undersampling_mask(
    image_size,
    acceleration=2
)

mask_r4 = create_undersampling_mask(
    image_size,
    acceleration=4
)


# ------------------------------------------------------------
# 4. Apply masks to k-space
# ------------------------------------------------------------

kspace_r2 = kspace * mask_r2

kspace_r4 = kspace * mask_r4


# ------------------------------------------------------------
# 5. Reconstruction function
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
# 6. Reconstruct images
# ------------------------------------------------------------

full_reconstruction = reconstruct(kspace)

reconstruction_r2 = reconstruct(kspace_r2)

reconstruction_r4 = reconstruct(kspace_r4)


# ------------------------------------------------------------
# 7. Display sampling masks
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))
plt.imshow(mask_r2, cmap="gray", aspect="auto")
plt.title("Cartesian Sampling Mask: R = 2")
plt.xlabel("Frequency-Encoding Direction")
plt.ylabel("Phase-Encoding Direction")
plt.tight_layout()


plt.figure(figsize=(6, 6))
plt.imshow(mask_r4, cmap="gray", aspect="auto")
plt.title("Cartesian Sampling Mask: R = 4")
plt.xlabel("Frequency-Encoding Direction")
plt.ylabel("Phase-Encoding Direction")
plt.tight_layout()


# ------------------------------------------------------------
# 8. Display reconstructed images
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))
plt.imshow(full_reconstruction, cmap="gray")
plt.title("Fully Sampled Reconstruction")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(reconstruction_r2, cmap="gray")
plt.title("Cartesian Undersampling: R = 2")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(reconstruction_r4, cmap="gray")
plt.title("Cartesian Undersampling: R = 4")
plt.axis("off")
plt.tight_layout()


plt.show()
