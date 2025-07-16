import os
from PIL import Image

def process_images(input_dir, output_dir, target_size=64, final_size=128):
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
    base_dir = os.path.join('data', 'vascular_function_128')
    input_dir = os.path.join(base_dir, 'images')
    output_dir = os.path.join(base_dir, 'images_processed')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process images
    process_images(input_dir, output_dir, target_size=40)


if __name__ == "__main__":
    main()

def test_single_image(size=64):
    """Test function to process a single image and display before/after."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "vascular_function_128", "images")
    output_dir = os.path.join(base_dir, "data", "vascular_function_128", "images_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first image that starts with 'C_'
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(image_extensions) and f.split('_')[0] == 'C']
    
    if not image_files:
        print("No matching image files found in the input directory (files should start with 'C_').")
        return
    
    filename = image_files[0]
    print(f"Testing with image: {filename}")
    
    # Process the image
    img_path = os.path.join(data_dir, filename)
    output_path = os.path.join(output_dir, f"test_{filename}")
    
    try:
        # Open and process the image
        img = Image.open(img_path)
        print(f"Original size: {img.size}")
        
        # Display original
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        
        # Crop to 32x32
        target_size = size
        width, height = img.size
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        cropped = img.crop((left, top, right, bottom))
        
        # Resize back to 128x128
        final_size = 128
        resized = cropped.resize((final_size, final_size), Image.LANCZOS)
        
        # Display result
        plt.subplot(1, 2, 2)
        plt.imshow(resized)
        plt.title("Processed (32x32 → 128x128)")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()


#test_single_image(40)

# try finding the max up, down, left, and right coords for all the images and crop each image to that size with a little bit of padding to make it a square and then expand to 128x128