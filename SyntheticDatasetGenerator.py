import os
import cv2
import numpy as np
import random

# --- Configuration ---
NUM_IMAGES = 500
IMG_SIZE = 640 # Standard YOLOv8 input size
CLASS_ID = 0   # '0' represents our 'beam' class

# Create YOLO directory structure
IMG_DIR = "synthetic_beam_data/images/train"
LABEL_DIR = "synthetic_beam_data/labels/train"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Pre-compute the meshgrid to save processing time
x_coords = np.linspace(0, IMG_SIZE - 1, IMG_SIZE)
y_coords = np.linspace(0, IMG_SIZE - 1, IMG_SIZE)
x, y = np.meshgrid(x_coords, y_coords)

def create_gaussian(x, y, amp, x0, y0, sig_x, sig_y):
    """Generates a 2D Gaussian array."""
    return amp * np.exp(-(((x - x0)**2) / (2 * sig_x**2) + ((y - y0)**2) / (2 * sig_y**2)))

def generate_beam_and_label(index):
    """Generates one image and its corresponding YOLO label."""
    
    # Randomly select the type of beam to generate
    beam_type = random.choices(
        ['normal', 'saturated', 'split'], 
        weights=[50, 25, 25], k=1
    )[0]

    # Base parameters for the primary beam
    x0 = random.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
    y0 = random.uniform(IMG_SIZE * 0.2, IMG_SIZE * 0.8)
    sig_x = random.uniform(15, 60)
    sig_y = random.uniform(15, 60)
    
    if beam_type == 'saturated':
        amp = random.uniform(300, 500) # Intentionally high to cause clipping at 255
    else:
        amp = random.uniform(100, 240) # Normal amplitude

    # Generate primary beam
    beam = create_gaussian(x, y, amp, x0, y0, sig_x, sig_y)
    
    # Initialize Bounding Box extremes based on 3-sigma (covers 99.7% of the beam)
    x_min = x0 - (3 * sig_x)
    x_max = x0 + (3 * sig_x)
    y_min = y0 - (3 * sig_y)
    y_max = y0 + (3 * sig_y)

    # Handle Split-Peak (Deformed Beam)
    if beam_type == 'split':
        # Add a second smaller, misaligned Gaussian nearby
        x0_2 = x0 + random.uniform(-50, 50)
        y0_2 = y0 + random.uniform(-50, 50)
        sig_x_2 = random.uniform(10, 30)
        sig_y_2 = random.uniform(10, 30)
        amp_2 = random.uniform(80, 200)
        
        beam_2 = create_gaussian(x, y, amp_2, x0_2, y0_2, sig_x_2, sig_y_2)
        beam = beam + beam_2 # Merge the two beams
        
        # Expand bounding box to include the second peak
        x_min = min(x_min, x0_2 - (3 * sig_x_2))
        x_max = max(x_max, x0_2 + (3 * sig_x_2))
        y_min = min(y_min, y0_2 - (3 * sig_y_2))
        y_max = max(y_max, y0_2 + (3 * sig_y_2))

    # Add background camera noise (Poisson/Gaussian mix approximation)
    noise = np.random.normal(5, 10, beam.shape)
    img_matrix = beam + noise
    
    # Clamp values to valid 8-bit image range [0, 255]
    img_matrix = np.clip(img_matrix, 0, 255).astype(np.uint8)

    # --- Calculate YOLO Normalized Coordinates ---
    # Ensure bounding box doesn't go outside image borders
    x_min, x_max = max(0, x_min), min(IMG_SIZE, x_max)
    y_min, y_max = max(0, y_min), min(IMG_SIZE, y_max)
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    box_x_center = x_min + (box_width / 2.0)
    box_y_center = y_min + (box_height / 2.0)
    
    # Normalize by dividing by image size
    norm_x_center = box_x_center / IMG_SIZE
    norm_y_center = box_y_center / IMG_SIZE
    norm_width = box_width / IMG_SIZE
    norm_height = box_height / IMG_SIZE

    # --- Save the Output ---
    img_filename = f"beam_{index:04d}_{beam_type}.jpg"
    label_filename = f"beam_{index:04d}_{beam_type}.txt"
    
    cv2.imwrite(os.path.join(IMG_DIR, img_filename), img_matrix)
    
    # Write the YOLO label text file
    with open(os.path.join(LABEL_DIR, label_filename), 'w') as f:
        f.write(f"{CLASS_ID} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

print("Generating synthetic dataset...")
for i in range(NUM_IMAGES):
    generate_beam_and_label(i)
print(f"Success! {NUM_IMAGES} images and labels created in 'synthetic_beam_data/'")