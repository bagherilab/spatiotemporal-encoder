#!/usr/bin/env python3
"""
Convert all PNG images in a directory to JPG, replacing transparency with black.

Usage:
    python convert_png_to_jpg.py input_dir [output_dir]

If output_dir is not provided, images will be saved in the same directory as input.
"""

import os
import sys
from pathlib import Path
from PIL import Image

def convert(input_path, output_path):
    """Convert a single PNG image to JPG, replacing transparency with black."""
    with Image.open(input_path) as img:
        # Convert to RGB if image has an alpha channel
        if img.mode in ('RGBA', 'LA'):
            # Create a white background image
            background = Image.new('RGB', img.size, (0, 0, 0))  # Black background
            # Paste the image on top, using the alpha channel as mask
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPG
        output_path = output_path.with_suffix('.jpg')
        img.save(output_path, 'JPEG', quality=95)
    return output_path

def process_directory(input_dir, output_dir=None):
    """Process all PNG files in the input directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all PNG files
    png_files = list(input_dir.glob('*.png'))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to convert...")
    
    for i, png_file in enumerate(png_files, 1):
        try:
            output_path = output_dir / png_file.name
            jpg_path = convert(png_file, output_path)
            print(f"[{i}/{len(png_files)}] Converted {png_file.name} -> {jpg_path}")
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
    
    print("\nConversion complete!")



process_directory('data/vf_256_processed/images', 'data/vf_256_processed_jpg/images')
