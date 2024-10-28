import cv2
import os
import re

# Directory containing images
directory = "results"

# Compile regex pattern to match filenames
pattern = re.compile(r'generation_0(\d+).png')

# Filter and sort image filenames based on "generation_0xy.png" pattern
images = sorted(
    [img for img in os.listdir(directory) if pattern.match(img)],
    key=lambda x: int(pattern.search(x).group(1))
)

# Check if images exist
if not images:
    raise ValueError("No images found in 'results' directory with the pattern 'generation_0xy.png'.")

# Read first image to set video properties
first_image = cv2.imread(os.path.join(directory, images[0]))
height, width, _ = first_image.shape
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

# Write each image to the video
for img_name in images:
    img_path = os.path.join(directory, img_name)
    img = cv2.imread(img_path)
    video.write(img)

# Release the video writer
video.release()
print("Video saved as 'output.mp4'")
