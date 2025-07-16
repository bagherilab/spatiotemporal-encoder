import os
from PIL import Image

def process_images(input_dir, output_dir, target_size=128, final_size=256):
    """
    Process images by cropping to target_size x target_size and then resizing to final_size x final_size.
    
    Args:
        input_dir (str): Path to the directory containing input images
        output_dir (str): Path to save processed images
        target_size (int): Size to crop the image to (square)
        final_size (int): Size to resize the cropped image to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory that start with 'C_'
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions) and f.split('_')[0] == 'C']
    
    if not image_files:
        print("No matching image files found in the input directory (files should start with 'C_').")
        return
        
    print(f"Found {len(image_files)} images to process.")
    
    for i, filename in enumerate(image_files, 1):
        try:
            # Open the image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            
            # Get dimensions
            width, height = img.size
            
            # Calculate coordinates to crop center
            left = (width - target_size) // 2
            top = (height - target_size) // 2
            right = left + target_size
            bottom = top + target_size
            
            # Crop and resize
            cropped = img.crop((left, top, right, bottom))
            resized = cropped.resize((final_size, final_size), Image.LANCZOS)
            
            # Save the processed image
            output_path = os.path.join(output_dir, filename)
            resized.save(output_path)
            print(f"Processed {i}/{len(image_files)}: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("\nImage processing complete!")


def main():
    # Define input and output directories
    input_dir = os.path.join('data', 'vf_256')
    output_dir = os.path.join('data', 'vf_256_processed')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process images
    # Process images with target_size=200 to give some margin for cropping
    process_images(input_dir, output_dir, target_size=128, final_size=128)


if __name__ == "__main__":
    main()

def test_single_image(size=128):
    """Test function to process a single image and display before/after.
    Tests on 'C_Lvav_49_150_cells_cancer.png' from vf_256 dataset.
    """
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Define paths
    data_dir = os.path.join('data', 'vf_256')
    output_dir = os.path.join('data', 'vf_256_processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find our specific test image
    target_image = 'C_Lvav_49_150_cells_cancer.png'
    
    try:
        # Load the specific test image
        img_path = os.path.join(data_dir, target_image)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Display original
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original ({img.size[0]}x{img.size[1]})")
        
        # Crop to target size (200x200 for vf_256)
        target_size = size
        width, height = img.size
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        cropped = img.crop((left, top, right, bottom))
        
        # Resize to final size (256x256 for vf_256)
        final_size = 128
        resized = cropped.resize((final_size, final_size), Image.LANCZOS)
        
        # Save the processed image
        output_path = os.path.join(output_dir, f"processed_{target_image}")
        resized.save(output_path)
        
        # Display result
        plt.subplot(1, 2, 2)
        plt.imshow(resized, cmap='gray')
        plt.title(f"Processed ({target_size}x{target_size} → {final_size}x{final_size})")
        
        plt.tight_layout()
        plt.show()
        
        # Print processing info
        print(f"Processed image saved to: {output_path}")
        print(f"Original size: {img.size}")
        print(f"Cropped to: {target_size}x{target_size}")
        print(f"Resized to: {final_size}x{final_size}")
        
    except FileNotFoundError:
        print(f"Error: Could not find test image at {os.path.join(data_dir, target_image)}")
        print("Please make sure the vf_256 dataset is properly set up in the data directory.")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()

# Uncomment to test
#test_single_image(128)

# try finding the max up, down, left, and right coords for all the images and crop each image to that size with a little bit of padding to make it a square and then expand to 128x128