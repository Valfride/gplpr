from pathlib import Path
from PIL import Image
import os
import shutil
import cv2
from matplotlib import pyplot as plt

import numpy as np

def generate_dataset(root, dest, file, dlk=(9, 9), loc=7, ):
    root = Path(root)
    dest = Path(dest)
    destHR = dest / Path('HR')
    destHR.mkdir(parents=True, exist_ok=True)
    destLR = dest / Path('LR')
    destLR.mkdir(parents=True, exist_ok=True)
    with open(root / Path(file), 'r', encoding='utf8') as tp:
        lines = tp.readlines()
    with open(dest / Path('split_all.txt'), 'w+', encoding='utf8') as fd:
        for line in tqdm(lines, total=len(lines)):
            # file, _set = line.replace('\n', '').split(';')
            file = line.replace('\n', '')
            _set = 'validation'
            with open(root / Path(file).with_suffix('.txt'), 'r', encoding='utf8') as fp:
                pts = fp.readlines()[loc].replace('\n', '').split(': ')[1].replace(',', ' ').split(' ')
                pts = np.array([np.array([int(x), int(y)], dtype='float32') for x, y in zip(*[iter(pts)]*2)])
                imgHR = cv2.imdecode(np.fromfile((root / Path(file)).as_posix(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # imgHR = cv2.imdecode(np.fromfile(np.fromfile((root / Path(file)).as_posix(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # imgHR = customDataset.rectify_img(customDataset, imgHR, pts)
                shutil.copy2((root / Path(file)).with_suffix('.txt'), destHR)
                
                imgLR = cv2.resize(imgHR, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_CUBIC)
                imgLR = cv2.GaussianBlur(imgLR, dlk, 0)
                
                tempHR = Path('tempHR.jpg')
                cv2.imwrite((destHR / tempHR).as_posix(), imgHR)
                os.rename((destHR / tempHR).as_posix(), (destHR / file.split('/')[-1]).as_posix())
                
                tempLR = Path('tempLR.jpg')
                cv2.imwrite((destLR / tempLR).as_posix(), imgLR)
                os.rename((destLR / tempLR).as_posix(), (destLR / file.split('/')[-1]).as_posix())
                
                fd.write((destHR / file.split('/')[-1]).as_posix()+';'+(destLR / file.split('/')[-1]).as_posix()+';'+_set+'\n')
                        

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
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
 
	return warped

def padding(img, min_ratio=1.85, max_ratio=2.15, color = (0, 0, 0)):
	img_h, img_w = np.shape(img)[:2]

	border_w = 0
	border_h = 0
	ar = float(img_w)/img_h

	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1
			ar = float(img_w+border_w)/(img_h+border_h)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

	border_w = border_w//2
	border_h = border_h//2

	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
    
	return img, border_w, border_h
# Define the source directory where your subfolders are located
source_directory = Path("./yj4Iu2-UFPR-ALPR/UFPR-ALPR dataset")

# Define the destination directory where you want to copy the files
destination_directory = Path("./UFPR_ALPR")

# Create the destination directory if it doesn't exist
# destination_directory.mkdir(parents=True, exist_ok=True)

destHR = destination_directory / Path('HR')
destHR.mkdir(parents=True, exist_ok=True)
destLR = destination_directory / Path('LR')
destLR.mkdir(parents=True, exist_ok=True)
# Create or open the "split_all.txt" file for writing
with open(os.path.join(destination_directory, "split_all.txt"), "w") as split_file:
    
    # Iterate through the subfolders: "training," "testing," and "validation"
    subfolders = ["training", "testing", "validation"]
    for subfolder in subfolders:
        subfolder_path = source_directory / subfolder

        # Recursively iterate through all subfolders and files
        for image_path in subfolder_path.rglob("*.png"):
            # Copy the image to the destination directory
            coor = []
            with open(image_path.with_suffix('.txt'), 'r') as f:
                lines = f.readlines()
                if 'motorcycle' not in lines[3]:
                    corners = lines[8].strip().split(' ')[1:]
                    for item in corners:
                        # print(image_path)
                        coor.append([int(i) for i in item.split(',')])
                    
                    hr = cv2.imread(image_path.as_posix())                    
                    hr = rectify_img(hr, np.array(coor, np.float32), margin=2)
                    lr = cv2.resize(hr, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_CUBIC)
                    lr = cv2.GaussianBlur(lr, (3, 3), 0)
                    
                    # img = padding(img, color=())
                    shutil.copy2(image_path.with_suffix('.txt'), destHR) 
                    image_destination_hr = destHR / image_path.name
                    image_destination_lr = destLR / image_path.name
                    
                    cv2.imwrite(image_destination_hr.as_posix(), hr)
                    cv2.imwrite(image_destination_lr.as_posix(), lr)
        
                    # Write the image filename and subfolder name to split_all.txt
                    split_file.write(f"{image_destination_hr.as_posix()};{image_destination_lr.as_posix()};{subfolder}\n")

print("Files copied and split_all.txt created successfully.")
