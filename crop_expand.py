import os
from PIL import Image

def process_images(input_dir, output_dir, target_size=32, final_size=128):
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
    
    # Get all image files in the input directory
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    for filename in image_files:
        try:
            # Open the image
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            
            # Get dimensions
            width, height = img.size
            
            # Calculate coordinates to crop center 32x32
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
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "vascular_function_128", "images")
    output_dir = os.path.join(base_dir, "data", "vascular_function_128", "images_processed")
    
    # Process images
    print(f"Processing images from: {data_dir}")
    print(f"Saving processed images to: {output_dir}")
    
    process_images(data_dir, output_dir)
    
    print("Image processing complete!")