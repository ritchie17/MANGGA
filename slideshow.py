import os
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip
import numpy as np

def resize_image(image, target_width, target_height):
    return image.resize((target_width, target_height), Image.LANCZOS)

def load_and_resize_images(folder_path, target_width, target_height, font_size, padding):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                image = resize_image(image, target_width, target_height)
                folder_name = os.path.basename(root)
                image = add_text_to_image(image, folder_name, font_size, padding)
                images.append(image)
    return images

def add_text_to_image(image, text, font_size, padding):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Try to use a TrueType font
    except IOError:
        font = ImageFont.load_default(font_size)  # Fallback to default font if TrueType font is not found

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    # Add padding to the text box
    text_position = (padding, padding)
    rect_position = [0, 0, image.width, text_size[1] + 2 * padding]
    draw.rectangle(rect_position, fill="black")
    draw.text(text_position, text, fill="white", font=font)
    return image

def images_to_mp4(images, output_path, display_time):
    # Convert PIL images to numpy arrays
    frames = [np.array(image.convert("RGB")) for image in images]
    # Create a video clip
    clip = ImageSequenceClip(frames, fps=1/display_time)
    clip.write_videofile(output_path, codec='libx264')

def main():
    folder_path = '/mnt/f/Clean Dataset for Copywrite/Segmented/Split 500 Non-Export 4'
    display_time = 0.2  # Set display time to 0.2 seconds
    output_path = '/mnt/f/Non-Export4.mp4'
    font_size = 60  # Adjust the font size as needed
    padding = 30  # Adjust the padding as needed

    target_width, target_height = 2304, 1296  # Resize images to 2304x1296
    images = load_and_resize_images(folder_path, target_width, target_height, font_size, padding)
    images_to_mp4(images, output_path, display_time)

if __name__ == "__main__":
    main()
