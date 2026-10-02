"""
Demonstration of complex Gaussian noise in MRI k-space.

This script creates a synthetic image, transforms it into k-space,
adds complex Gaussian noise at two different levels, and reconstructs
magnitude images using the inverse Fourier transform.

The example demonstrates how noise in acquired k-space data affects
the reconstructed MRI image.

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
# 2. Transform image into centered k-space
# ------------------------------------------------------------

kspace = np.fft.fftshift(
    np.fft.fft2(
        np.fft.ifftshift(image)
    )
)


# ------------------------------------------------------------
# 3. Add complex Gaussian noise to k-space
# ------------------------------------------------------------

def add_complex_noise(kspace_data, noise_std, rng):
    """
    Add independent Gaussian noise to the real and imaginary
    components of k-space.
    """

    noise_real = rng.normal(
        0,
        noise_std,
        kspace_data.shape
    )

    noise_imag = rng.normal(
        0,
        noise_std,
        kspace_data.shape
    )

    complex_noise = noise_real + 1j * noise_imag

    return kspace_data + complex_noise


# Reproducible random-number generator
rng = np.random.default_rng(42)

low_noise_std = 5
high_noise_std = 20

kspace_low_noise = add_complex_noise(
    kspace,
    low_noise_std,
    rng
)

kspace_high_noise = add_complex_noise(
    kspace,
    high_noise_std,
    rng
)


# ------------------------------------------------------------
# 4. Reconstruction function
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
# 5. Reconstruct images
# ------------------------------------------------------------

reconstruction_clean = reconstruct(kspace)

reconstruction_low_noise = reconstruct(
    kspace_low_noise
)

reconstruction_high_noise = reconstruct(
    kspace_high_noise
)


# ------------------------------------------------------------
# 6. Calculate simple image-domain SNR estimates
# ------------------------------------------------------------

# Signal ROI: central part of the main object
signal_roi = (
    (x - 64) ** 2
    + (y - 64) ** 2
    <= 15 ** 2
)

# Background ROI: upper-left corner
background_roi = (
    (x < 20)
    & (y < 20)
)


def estimate_snr(image_data):
    """
    Simple illustrative SNR estimate:
    mean signal in object ROI divided by standard deviation
    in a background ROI.

    This is used only for demonstration and is not intended as
    a general MRI SNR measurement method.
    """

    mean_signal = np.mean(
        image_data[signal_roi]
    )

    noise_std = np.std(
        image_data[background_roi]
    )

    return mean_signal / noise_std


snr_low = estimate_snr(
    reconstruction_low_noise
)

snr_high = estimate_snr(
    reconstruction_high_noise
)


print(
    f"Estimated SNR - low noise: "
    f"{snr_low:.2f}"
)

print(
    f"Estimated SNR - high noise: "
    f"{snr_high:.2f}"
)


# ------------------------------------------------------------
# 7. Visualize reconstructed images
# ------------------------------------------------------------

display_max = 2.0

plt.figure(figsize=(5, 5))
plt.imshow(
    reconstruction_clean,
    cmap="gray",
    vmin=0,
    vmax=display_max
)
plt.title("Noise-Free Reconstruction")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(
    reconstruction_low_noise,
    cmap="gray",
    vmin=0,
    vmax=display_max
)
plt.title("Low K-space Noise")
plt.axis("off")
plt.tight_layout()


plt.figure(figsize=(5, 5))
plt.imshow(
    reconstruction_high_noise,
    cmap="gray",
    vmin=0,
    vmax=display_max
)
plt.title("High K-space Noise")
plt.axis("off")
plt.tight_layout()


plt.show()
