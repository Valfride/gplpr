import os
import re
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

def extract_corners_from_file(file_path):
    """
    Reads a text file, extracts the corner points using regular expressions, and returns them as a numpy array.
    
    Parameters:
        file_path (str or Path): The path to the text file containing the corners information.
        
    Returns:
        np.ndarray: A numpy array of shape (4, 2) containing the corner points.
    """
    # Read the content of the file
    with open(file_path, 'r') as file:
        text = file.read()

    # Regular expression to find the corners
    match = re.search(r"corners:\s*(\d+,\d+\s+\d+,\d+\s+\d+,\d+\s+\d+,\d+)", text)
    
    if match:
        # Extract the matched string
        points_str = match.group(1)
        
        # Split the points into pairs and convert to numpy array
        points = np.array([list(map(int, pair.split(','))) for pair in points_str.split()])
        
        return points
    else:
        raise ValueError("Corners not found in the provided text")

def crop_license_plate(image, points, margin=2):
    """
    Crop the license plate from the image using the bounding box around the given points.
    
    Parameters:
        image (numpy.ndarray): The original image.
        points (numpy.ndarray): The coordinates of the four corners of the license plate in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        margin (int): The margin to add around the bounding box. Default is 2 pixels.
        
    Returns:
        cropped_image (numpy.ndarray): The cropped image containing the license plate.
        new_points (numpy.ndarray): The coordinates of the points relative to the cropped image.
    """
    # Calculate the bounding box (axis-aligned)
    x, y, w, h = cv2.boundingRect(points)

    # Add the margin to the bounding box
    x_min = max(x - margin, 0)
    y_min = max(y - margin, 0)
    x_max = min(x + w + margin, image.shape[1])
    y_max = min(y + h + margin, image.shape[0])

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Relabel the new points relative to the cropped image
    new_points = points - [x_min, y_min]

    return cropped_image, new_points

def crop_license_plate_with_padding(image, points, width_percentage=0.1, height_percentage=0.1):
    """
    Crops the license plate with padding based on a percentage of its width and height.
    
    Parameters:
        image (np.ndarray): The image containing the license plate.
        points (np.ndarray): The four corner points of the license plate.
        width_percentage (float): Percentage of the width to add as padding.
        height_percentage (float): Percentage of the height to add as padding.
        
    Returns:
        cropped_image (np.ndarray): The cropped image with padding.
        new_points (np.ndarray): The new corner points relative to the cropped image.
    """
    # Calculate the bounding box around the points
    x, y, w, h = cv2.boundingRect(points)
    
    # Calculate padding based on percentage of width and height
    x_padding = int(w * width_percentage)
    y_padding = int(h * height_percentage)
    
    # Calculate new bounding box with padding
    x_min = max(x - x_padding, 0)
    y_min = max(y - y_padding, 0)
    x_max = min(x + w + x_padding, image.shape[1])
    y_max = min(y + h + y_padding, image.shape[0])
    
    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Relabel the points relative to the new cropped image
    new_points = points - [x_min, y_min]
    
    return cropped_image, new_points

def rectify_img(img, pts, margin=2):
        # obtain a consistent order of the points and unpack them individually
        # rect = order_points(pts)
        (tl, tr, br, bl) = pts

        # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        maxWidth += margin*2
        maxHeight += margin*2

        # now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
        ww = maxWidth - 1 - margin
        hh = maxHeight - 1 - margin
        c1 = [margin, margin]
        c2 = [ww, margin]
        c3 = [ww, hh]
        c4 = [margin, hh]

        dst = np.array([c1, c2, c3, c4], dtype = 'float32')

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped, dst

def update_and_save_text_file(original_file_path, dest_path, new_points):
    """
    Updates the text file with new corner points and saves it in the destination folder.
    
    Parameters:
        original_file_path (str or Path): The path to the original text file.
        new_points (np.ndarray): The new corner points to write to the file.
    """
    with open(original_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the line with "corners:" and replace it with the updated points
    updated_lines = []
    for line in lines:
        if line.startswith("corners:"):
            points_str = ' '.join([f"{x},{y}" for x, y in new_points])
            updated_lines.append(f"corners: {points_str}\n")
        else:
            updated_lines.append(line)
    
    # Write the updated lines to the new text file
    updated_file_path = dest_path.with_suffix('.txt')
    with open(updated_file_path, 'w') as file:
        file.writelines(updated_lines)

def draw_polygon_on_image(image, points, color=(0, 255, 0), thickness=1):
    """
    Draws a polygon connecting the points on the image.
    
    Parameters:
        image (np.ndarray): The image to draw the polygon on.
        points (np.ndarray): The points to connect.
        color (tuple): The color of the polygon in BGR format.
        thickness (int): The thickness of the polygon border.
        
    Returns:
        np.ndarray: The image with the polygon drawn on it.
    """
    # Ensure points are in the right format (int and shape of (n, 1, 2))
    points = points.reshape((-1, 1, 2))
    
    # Draw the polygon
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    
    plt.imshow(image)
    plt.show()

def replace_prefix_in_file(file_path, old_prefix, new_prefix):
    """
    Replaces a specific prefix in each line of a text file with a new prefix.
    
    Parameters:
        file_path (str or Path): The path to the text file.
        old_prefix (str): The old prefix to be replaced.
        new_prefix (str): The new prefix to replace the old prefix.
    """
    from pathlib import Path
    
    file_path = Path(file_path)
    
    # Read the original file content
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Replace old_prefix with new_prefix in each line
    new_lines = [line.replace(old_prefix, new_prefix) for line in lines]
    print(Path(new_lines[0].split(';')[0]).is_file())
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

def process_images_in_folders(folders, output_folder="Rodosol-processed", if_rectify=False, if_percentage=True):
    output_folder = Path(output_folder)
    
    for folder in folders:
        folder_path = Path(folder)
        image_paths = list(folder_path.rglob('*'))
        image_paths = [img_path for img_path in image_paths if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]
        
        # Traverse the folder
        for img_path in tqdm(image_paths, desc=f"Processing images in {folder_path}", unit="image"):
            # Load the image
            image = cv2.imread(str(img_path))
            txt_path = img_path.with_suffix('.txt')
            # Example: you would need to provide actual points
            # In practice, you need to have a way to get these points for each image
            points = extract_corners_from_file(txt_path)  # Replace with actual points
            
            # Process the image
            if if_percentage:
                cropped_image, new_points = crop_license_plate_with_padding(image, points)
            else:
                cropped_image, new_points = crop_license_plate(image, points)
            # draw_polygon_on_image(cropped_image, new_points)
            if if_rectify is True:
               cropped_image, new_points = rectify_img(cropped_image, new_points)
            # draw_polygon_on_image(cropped_image, new_points.astype(np.int32))  
            # Construct the output path
            relative_path = img_path.relative_to(folder_path.parent)
            output_path = output_folder / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the cropped image
            cv2.imwrite(str(output_path), cropped_image)
            update_and_save_text_file(txt_path, output_path.with_suffix('.txt'), new_points)
                    
                # print(f"Processed and saved: {output_path}")

if __name__ == '__main__':    
    # Example usage:r"./Rodosol/cars-br", r"./Rososol/cars-me", r"./Rososol/motorcycles-br",
    # folders_to_process = [r"./Rodosol/cars-me", r"./Rodosol/cars-br", r"./Rodosol/motorcycles-br", r"./Rodosol/motorcycles-me"]
    # process_images_in_folders(folders_to_process, output_folder="Rodosol-cropped_not_rect")
    # process_images_in_folders(folders_to_process, output_folder="Rodosol-cropped_rect", if_rectify=True)
    
    # replace_prefix_in_file(r"./Rodosol-cropped_not_rect/split.txt", "./images", "./Rodosol-cropped_not_rect")
    # replace_prefix_in_file(r"./Rodosol-cropped_rect/split.txt", "./images", "./Rodosol-cropped_not_rect")
    
    # folders_to_process = [r"./Rodosol/cars-me", r"./Rodosol/cars-br", r"./Rodosol/motorcycles-br", r"./Rodosol/motorcycles-me"]
    # process_images_in_folders(folders_to_process, output_folder="Rodosol-cropped_not_rect_percentage")
    # process_images_in_folders(folders_to_process, output_folder="Rodosol-cropped_rect_percentage", if_rectify=True)
    
    replace_prefix_in_file(r"./Rodosol-cropped_not_rect_percentage/split.txt", "./images", "./Rodosol-cropped_not_rect_percentage")
    replace_prefix_in_file(r"./Rodosol-cropped_rect_percentage/split.txt", "./images", "./Rodosol-cropped_rect_percentage")
    