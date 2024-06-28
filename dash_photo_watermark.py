import cv2

# dashcam
# mio

# opencv mask

# Function to display the selected pixel color
def display_pixel_color(x, y):
    pixel = cropped_image[y, x]
    rgb_color = (pixel[2], pixel[1], pixel[0])
    print(f"RGB Color at ({x}, {y}): {rgb_color}")

# Function to get a zoomed-in view around a single pixel
def get_zoomed_image(image, cursor_x, cursor_y, zoom_factor=500):
    pixel = image[cursor_y, cursor_x]
    zoomed_image = cv2.resize(pixel.reshape(1, 1, 3), (zoom_factor, zoom_factor), interpolation=cv2.INTER_NEAREST)
    return zoomed_image


# Image path
image_path = r'D:\Python_Projects\Geoguessr_tw\dataset\Changhua County\3V2euty7TrCTbJdUroQv4H\134407865348048.jpg'

# Load an image
image = cv2.imread(image_path)

# Get the original image dimensions
orig_height, orig_width = image.shape[:2]

# Get the original image dimensions
orig_height, orig_width = image.shape[:2]

# Define the maximum width and height for cropping
max_width = 800  # Adjust as needed
max_height = 600  # Adjust as needed

# Calculate the coordinates for the bottom-right crop
crop_x = max(0, orig_width - max_width)
crop_y = max(0, orig_height - max_height)

# Crop the image
cropped_image = image[crop_y:orig_height, crop_x:orig_width]

# Get the dimensions of the cropped image
cropped_height, cropped_width = cropped_image.shape[:2]

# Initialize cursor position
cursor_x = cropped_width // 2
cursor_y = cropped_height // 2

cv2.namedWindow('Image')
cv2.namedWindow('Zoomed Image')

cursor_radius = 2

while True:
    # Copy the image to draw the cursor
    display_image = cropped_image.copy()

    # Draw a red circle at the cursor position
    cv2.circle(display_image, (cursor_x, cursor_y), cursor_radius, (0, 0, 255), -1)

    # Get the zoomed-in image around the cursor
    zoomed_image = get_zoomed_image(cropped_image, cursor_x, cursor_y)

    # Display the image with the cursor
    cv2.imshow('Image', display_image)

    # Display the zoomed-in view
    cv2.imshow('Zoomed Image', zoomed_image)

    # Wait for a key press
    key = cv2.waitKey(0)

    # Move the cursor based on the key pressed
    if key == ord('q'):
        break
    elif key == ord('w'):  # Up
        cursor_y = max(cursor_y - 1, 0)
    elif key == ord('s'):  # Down
        cursor_y = min(cursor_y + 1, cropped_height - 1)
    elif key == ord('a'):  # Left
        cursor_x = max(cursor_x - 1, 0)
    elif key == ord('d'):  # Right
        cursor_x = min(cursor_x + 1, cropped_width - 1)

    # Display the pixel color at the new cursor position
    display_pixel_color(cursor_x, cursor_y)

# Release the resources
cv2.destroyAllWindows()