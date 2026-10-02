# MRI Reconstruction Basics

This repository contains basic Python examples for learning and demonstrating fundamental concepts in MRI reconstruction.

The examples use synthetic data to illustrate the relationship between image space and k-space, spatial-frequency information, Cartesian undersampling, complex k-space noise, and multi-coil image reconstruction.

## Examples

### 1. Basic Fourier Reconstruction

`fft_reconstruction.py`

Demonstrates:

- Creation of a synthetic image
- 2D Fourier transformation from image space to k-space
- Visualization of k-space using log magnitude
- Inverse Fourier reconstruction
- Numerical comparison between the original and reconstructed images

### 2. K-space Sampling

`kspace_sampling.py`

Demonstrates:

- Central and peripheral regions of k-space
- Low and high spatial-frequency information
- Image reconstruction using central k-space only
- Image reconstruction using peripheral k-space only
- Effects of spatial-frequency content on image detail and blurring

### 3. Cartesian Undersampling

`cartesian_undersampling.py`

Demonstrates:

- Regular Cartesian k-space undersampling
- Acceleration factors R = 2 and R = 4
- Undersampling along the phase-encoding direction
- Aliasing caused by insufficient k-space sampling
- Difference between loss of high spatial frequencies and regular undersampling

### 4. K-space Noise

`kspace_noise.py`

Demonstrates:

- Complex Gaussian noise in k-space
- Independent noise in real and imaginary signal components
- Reconstruction of noisy magnitude images
- Comparison of low- and high-noise conditions
- Simple illustrative image-domain SNR estimation

### 5. Multi-Coil Reconstruction

`multicoil_reconstruction.py`

Demonstrates:

- Synthetic receiver-coil sensitivity profiles
- Coil-specific image formation
- Coil-specific k-space data
- Reconstruction of individual receiver-coil images
- Root-sum-of-squares (RSS) coil combination
- Spatial intensity variation associated with coil sensitivities

### 6. Basic SENSE Reconstruction

`basic_sense_reconstruction.py`

Demonstrates:

- R = 2 image-domain folding
- Spatial encoding using multiple receiver coils
- Coil sensitivity matrices
- Separation of overlapping spatial locations
- SENSE unfolding using the Moore-Penrose pseudoinverse
- Comparison of aliased and reconstructed images

This is a simplified educational implementation using known synthetic
coil sensitivity maps rather than a clinical SENSE reconstruction pipeline.


## Concepts Covered

The repository currently covers:

- Image space and k-space
- 2D FFT and inverse FFT
- Low and high spatial frequencies
- Cartesian k-space sampling
- Aliasing and image blurring
- Complex-valued MRI data
- K-space noise and image SNR
- Receiver-coil sensitivity
- Multi-coil MRI reconstruction
- Root-sum-of-squares coil combination
- Parallel imaging fundamentals
- SENSE reconstruction
- Coil sensitivity encoding
- Pseudoinverse-based unfolding

## Requirements

- Python 3
- NumPy
- Matplotlib

## Data

All examples use synthetic data. No patient data, identifiable information, restricted research data, or proprietary scanner code are included.

## Scope

These examples are intended to demonstrate fundamental MRI reconstruction concepts and are not implementations of vendor-specific or clinical reconstruction pipelines.

## Author

Yi Li  
Doctoral Researcher  
Research Unit of Health Sciences and Technology  
University of Oulu, Finland
Doctoral Researcher  
Research Unit of Health Sciences and Technology  
University of Oulu, Finland
