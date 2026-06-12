# Synchrotron Beam Synthetic Dataset Generator

This repository contains a numerical simulation script to procedurally generate a custom object-detection dataset for fine-tuning YOLOv8 models to detect focused synchrotron beam spots.

## Dataset Structure
The script automatically generates 500 images including:
- **Normal Beams:** Pristine 2D Gaussian distributions.
- **Saturated Beams:** Simulating camera sensor clipping at an 8-bit peak threshold of 255.
- **Split-Peak Beams:** Simulating beamline mirror aberrations or distorted profiles.

## How to Generate the Dataset Locally

1. Install dependencies:
   ```bash
   pip install opencv-python numpy
