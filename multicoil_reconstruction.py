"""
Basic multi-coil MRI reconstruction using synthetic coil sensitivities.

This script demonstrates:

1. Creation of a synthetic object
2. Simulation of four receiver-coil sensitivity profiles
3. Generation of coil-specific images and k-space data
4. Reconstruction of individual coil images
5. Root-sum-of-squares (RSS) coil combination

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

image[region_1] = 1.0
image[region_2] = 1.8


# ------------------------------------------------------------
# 2. Create four synthetic coil sensitivity profiles
# ------------------------------------------------------------

def gaussian_sensitivity(x, y, center_x, center_y, sigma):
    """
    Create a smooth Gaussian coil sensitivity profile.
    """

    return np.exp(
        -(
            (x - center_x) ** 2
            + (y - center_y) ** 2
        )
        / (2 * sigma ** 2)
    )


sigma = 55

coil_centers = [
    (20, 20),
    (108, 20),
    (20, 108),
    (108, 108),
]

coil_sensitivities = []

for center_x, center_y in coil_centers:
    sensitivity = gaussian_sensitivity(
        x,
        y,
        center_x,
        center_y,
        sigma
    )

    coil_sensitivities.append(sensitivity)

coil_sensitivities = np.array(
    coil_sensitivities
)


# ------------------------------------------------------------
# 3. Generate coil-specific images
# ------------------------------------------------------------

coil_images = (
    coil_sensitivities
    * image[np.newaxis, :, :]
)


# ------------------------------------------------------------
# 4. Transform each coil image into k-space
# ------------------------------------------------------------

coil_kspace = np.zeros(
    coil_images.shape,
    dtype=complex
)

for coil in range(
    coil_images.shape[0]
):
    coil_kspace[coil] = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(
                coil_images[coil]
            )
        )
    )


# ------------------------------------------------------------
# 5. Reconstruct individual coil images
# ------------------------------------------------------------

reconstructed_coils = np.zeros(
    coil_images.shape,
    dtype=complex
)

for coil in range(
    coil_kspace.shape[0]
):
    reconstructed_coils[coil] = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(
                coil_kspace[coil]
            )
        )
    )


# ------------------------------------------------------------
# 6. Combine coils using root-sum-of-squares
# ------------------------------------------------------------

rss_image = np.sqrt(
    np.sum(
        np.abs(reconstructed_coils) ** 2,
        axis=0
    )
)


# Normalize for visualization
rss_normalized = (
    rss_image
    / np.max(rss_image)
)

image_normalized = (
    image
    / np.max(image)
)


# ------------------------------------------------------------
# 7. Display coil sensitivity profiles
# ------------------------------------------------------------

for coil in range(
    coil_sensitivities.shape[0]
):
    plt.figure(figsize=(5, 5))

    plt.imshow(
        coil_sensitivities[coil],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Coil {coil + 1} Sensitivity"
    )

    plt.axis("off")
    plt.tight_layout()


# ------------------------------------------------------------
# 8. Display individual coil reconstructions
# ------------------------------------------------------------

for coil in range(
    reconstructed_coils.shape[0]
):
    plt.figure(figsize=(5, 5))

    plt.imshow(
        np.abs(
            reconstructed_coils[coil]
        ),
        cmap="gray"
    )

    plt.title(
        f"Coil {coil + 1} Reconstruction"
    )

    plt.axis("off")
    plt.tight_layout()


# ------------------------------------------------------------
# 9. Display original and RSS-combined images
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))

plt.imshow(
    image_normalized,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Original Synthetic Image"
)

plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))

plt.imshow(
    rss_normalized,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "RSS Multi-Coil Reconstruction"
)

plt.axis("off")
plt.tight_layout()


plt.show()
