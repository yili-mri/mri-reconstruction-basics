"""
Basic SENSE reconstruction using synthetic coil sensitivity maps.

This script demonstrates the core principle of SENSE parallel imaging:

1. Create a synthetic object
2. Create four spatially varying receiver-coil sensitivities
3. Simulate R = 2 aliasing along the phase-encoding direction
4. Use coil sensitivity information to separate overlapping pixels
5. Compare aliased and SENSE-reconstructed images

This is a simplified educational implementation using synthetic data.
It is not a clinical or vendor-specific SENSE reconstruction pipeline.
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

# Off-center object to make folding easier to see
region_3 = (
    (x - 25) ** 2
    + (y - 100) ** 2
    <= 9 ** 2
)

image[region_1] = 1.0
image[region_2] = 1.8
image[region_3] = 1.4


# ------------------------------------------------------------
# 2. Create four synthetic coil sensitivity profiles
# ------------------------------------------------------------

def gaussian_sensitivity(x, y, center_x, center_y, sigma):
    """Create a smooth Gaussian receiver-coil sensitivity profile."""

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

    coil_sensitivities.append(
        sensitivity
    )


coil_sensitivities = np.array(
    coil_sensitivities
)


# ------------------------------------------------------------
# 3. Generate full-FOV coil images
# ------------------------------------------------------------

coil_images = (
    coil_sensitivities
    * image[np.newaxis, :, :]
)


# ------------------------------------------------------------
# 4. Simulate R = 2 folding
# ------------------------------------------------------------

acceleration = 2

folded_size = (
    image_size // acceleration
)

# For R = 2, pixels separated by half the FOV
# overlap in the reduced-FOV image.
folded_coils = (
    coil_images[:, :folded_size, :]
    +
    coil_images[:, folded_size:, :]
)


# ------------------------------------------------------------
# 5. Display RSS image of the aliased coil data
# ------------------------------------------------------------

aliased_rss = np.sqrt(
    np.sum(
        np.abs(folded_coils) ** 2,
        axis=0
    )
)


# ------------------------------------------------------------
# 6. Perform basic SENSE unfolding
# ------------------------------------------------------------

sense_reconstruction = np.zeros(
    (image_size, image_size),
    dtype=complex
)


for row in range(folded_size):

    paired_row = row + folded_size

    for col in range(image_size):

        # Measurements from all four coils
        signal_vector = folded_coils[
            :,
            row,
            col
        ]

        # Sensitivity matrix:
        # column 1 -> first spatial location
        # column 2 -> folded spatial location
        sensitivity_matrix = np.column_stack(
            (
                coil_sensitivities[
                    :,
                    row,
                    col
                ],
                coil_sensitivities[
                    :,
                    paired_row,
                    col
                ]
            )
        )

        # Solve:
        #
        # signal = sensitivity_matrix @ unknown_pixels
        #
        # using the Moore-Penrose pseudoinverse.
        unfolded_pixels = (
            np.linalg.pinv(
                sensitivity_matrix
            )
            @ signal_vector
        )

        sense_reconstruction[
            row,
            col
        ] = unfolded_pixels[0]

        sense_reconstruction[
            paired_row,
            col
        ] = unfolded_pixels[1]


sense_magnitude = np.abs(
    sense_reconstruction
)


# ------------------------------------------------------------
# 7. Calculate reconstruction error
# ------------------------------------------------------------

mean_absolute_error = np.mean(
    np.abs(
        image
        - sense_magnitude
    )
)

print(
    "Mean absolute SENSE reconstruction error: "
    f"{mean_absolute_error:.6e}"
)


# ------------------------------------------------------------
# 8. Normalize images for visualization
# ------------------------------------------------------------

original_normalized = (
    image
    / np.max(image)
)

aliased_normalized = (
    aliased_rss
    / np.max(aliased_rss)
)

sense_normalized = (
    sense_magnitude
    / np.max(sense_magnitude)
)


# ------------------------------------------------------------
# 9. Visualize results
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))
plt.imshow(
    original_normalized,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Original Synthetic Image")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(
    aliased_normalized,
    cmap="gray",
    vmin=0,
    vmax=1,
    aspect="auto"
)
plt.title("R = 2 Aliased Multi-Coil Image")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(
    sense_normalized,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Basic SENSE Reconstruction")
plt.axis("off")
plt.tight_layout()


plt.show()
