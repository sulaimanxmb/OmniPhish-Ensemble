import sys
import os
from PIL import Image

def convert_image_to_pdf(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find '{image_path}'")
        print("Please make sure you saved the image there!")
        return

    # The paper expects it in the visualizations/ folder at the root!
    # Get the path to the visualizations folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    output_folder = os.path.join(project_root, 'visualizations')
    output_path = os.path.join(output_folder, 'Architecture.pdf')

    # Ensure the visualizations folder exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading image from: {image_path}")
    # Open image, convert to RGB (removes alpha channel which PDF doesn't support well)
    img = Image.open(image_path).convert('RGB')
    
    # Save as high quality PDF
    img.save(output_path, "PDF", resolution=100.0)
    print(f"SUCCESS! Successfully converted and saved to: {output_path}")
    print("Your Overleaf paper will now compile perfectly!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_arch.py <path_to_your_image.png>")
    else:
        convert_image_to_pdf(sys.argv[1])
