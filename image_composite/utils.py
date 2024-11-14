import cv2
import numpy as np
import os
import time


def draw_bbox(img, bbox, color, thickness:int):
    """
    Parameters
    ----------
    img
        d
    bbox: int array
        [[x1 y1] [x2 y2]]
    """
    drawed_img = cv2.rectangle(img.copy(), pt1=bbox[0], pt2=bbox[1], color=color, thickness=thickness)
    return drawed_img


def crop_nonzero(arr):
    # Find the indices of non-zero values
    nonzero_indices = np.argwhere(arr != 0)
    
    if nonzero_indices.size == 0:
        return arr  # Return the original array if there are no non-zero values
    
    # Calculate the coordinates of the rectangle
    top_left = nonzero_indices.min(axis=0)
    bottom_right = nonzero_indices.max(axis=0) + 1  # Add 1 to include the last index

    # Crop the rectangle
    return arr[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


def fit_rotate(img, scale=1.0, angle=0):
    # Translate the image
    (h, w) = img.shape[:2]
    r = 1.8 * max(h,w) * scale # new center
    M = np.array([[1,0, r-w/2],
                  [0,1, r-h/2]], dtype=np.float64)
    translated_img = cv2.warpAffine(img, M, (int(2*r), int(2*r)), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Rotate the image
    M = cv2.getRotationMatrix2D((r,r), angle, scale)
    rotated_img = cv2.warpAffine(translated_img, M, (int(2*r), int(2*r)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    cropped_img = crop_nonzero(rotated_img)
    return cropped_img


def resize(img, scale):

    height, width = img.shape[:2]

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_img


def overlay_image(background, bg_scale, object_img, x, y, obj_scale=1.0, angle=0,
                  bg_weight=0.8, obj_weight=1, gamma=0):
    """
    For X-ray images

    Returns
    -------
    combined_img, background, object_img: np.ndarray
        Images after revision
    """

    background = resize(background, scale=bg_scale)

    # Rotate and resize the object image
    object_img = fit_rotate(img=object_img, scale=obj_scale, angle=angle)

    # Get dimensions after rotation
    h, w = object_img.shape[:2]

    # Define region in the background to overlay the object
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    assert 0 < y1 < y2 < background.shape[0], f"object_img.shape={object_img.shape}, background.shape={background.shape}\ny1={y1},y2={y2}"
    assert 0 < x1 < x2 < background.shape[1], f"object_img.shape={object_img.shape}, background.shape={background.shape}\nx1={x1},x2={x2}"

    # Make the shape of the object image to be same as background.
    M = np.array([[1,0, x1],
                  [0,1, y1]], dtype=np.float64)
    margined_object_img = cv2.warpAffine(object_img, M, background.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Add the two images
    combined_img = cv2.addWeighted(src1=background ,alpha=bg_weight, src2=margined_object_img, beta=obj_weight, gamma=gamma)
    combined_img = 255 - combined_img
    return combined_img, background, object_img



def create_yolo_label(image, bbox, image_filename, label_filename, class_id=0):
    """Saves image and YOLO formatted label file for YOLO training.

    Parameters
    ----------
    - image: The image array.
    - bbox: Bounding box in (x_min, y_min, x_max, y_max) format.
    - image_filename: File path to save the image.
    - label_filename: File path to save the label file.
    - class_id: Integer class ID for the object (default is 0).
    """
    # Save image
    cv2.imwrite(image_filename, image)
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Calculate YOLO format bounding box coordinates
    [x_min, y_min], [x_max, y_max] = bbox
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Write YOLO label to file
    with open(label_filename, 'w') as label_file:
        label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


class GenYoloData():
    def __init__(self, bg_filename, obj_filename,
                 ):
        # Load background and object images
        self.bg_filename = bg_filename
        self.obj_filename = obj_filename
        self.background = cv2.imread(bg_filename, cv2.IMREAD_GRAYSCALE)
        self.object_img = cv2.imread(obj_filename, cv2.IMREAD_GRAYSCALE)
        assert self.background is not None, f"Wrong filename. bg_filename={bg_filename}"
        assert self.object_img is not None, f"Wrong filename. obj_filename={obj_filename}"
        self.bbox = None
        self.combined_img = None

    def combine(self, bg_scale, obj_scale, obj_range,
                bg_weight, obj_weight, gamma,
                regionimg_filename=None):
        obj_range = np.array(obj_range, dtype=int)

        # Calculate variables
        obj_origin_range = obj_range.astype(float)
        obj_origin_range[1] -= np.linalg.norm(self.object_img.shape[:2]) * (obj_scale/bg_scale)
        obj_origin_range = obj_origin_range.astype(int)
        angle = np.random.uniform(0,360)
        x = int(np.random.uniform(*obj_origin_range[:,0]) * bg_scale)
        y = int(np.random.uniform(*obj_origin_range[:,1]) * bg_scale)
        obj_origin = np.array([x,y], dtype=int)

        # Combine images
        combined_img, background, object_img = overlay_image(background=self.background, bg_scale=bg_scale, 
                                                             object_img=self.object_img, obj_scale=obj_scale, x=x, y=y, angle=angle,
                                                             bg_weight=bg_weight, obj_weight=obj_weight, gamma=gamma,
                                                             )
        self.combined_img = combined_img

        # Calculate bounding box
        self.bbox = np.zeros((2,2)).astype(int) # [[x1 y1] [x2 y2]]
        self.bbox[0] = obj_origin
        self.bbox[1] = obj_origin + np.array(object_img.shape[:2][::-1])

        if isinstance(regionimg_filename, str):
            # Verify the range
            bg_range_img = draw_bbox(background, bbox=obj_range, color=50, thickness=2)
            bg_range_img = draw_bbox(bg_range_img, bbox=obj_origin_range, color=200, thickness=2)
            cv2.imwrite(regionimg_filename, bg_range_img)
        return self.combined_img
    def save(self, directory, dataname, class_id=0):
        image_filename = directory + '/images/' + dataname + '.jpg'
        label_filename = directory + '/labels/' + dataname + '.txt'
        assert not os.path.exists(image_filename), f"The file already exists: {image_filename}"
        assert not os.path.exists(label_filename), f"The file already exists: {label_filename}"
        os.makedirs(os.path.dirname(image_filename), exist_ok=True)
        os.makedirs(os.path.dirname(label_filename), exist_ok=True)
        create_yolo_label(image=self.combined_img, bbox=self.bbox, image_filename=image_filename, label_filename=label_filename, class_id=class_id)
        self.bbox = None
        self.combined_img = None



if __name__ == "__main__":
    start_main = time.time()
    genyolo = GenYoloData(bg_filename='raw_images/background/ddca3f92-4b8e-4672-bb6b-f3594ad4e304.jpg',
                          obj_filename='raw_images/object/(ry0)(rx20)(delx0.1).png')
    for i in range(3):
        start_individual = time.time()
        dataname = f'data{i}'
        genyolo.combine(bg_scale=8.0, obj_scale=0.8, obj_range=[[120,120],  # x1 y1
                                                                [470,450]], # x2 y2
                        bg_weight=0.6, obj_weight=1.3, gamma=0.,
                        )
        genyolo.save(directory='Dataset.yolo/unannotated/', dataname=dataname)
        print(f"{i}th data saved: {dataname}")
        print("    Individual consumed time:", time.time()-start_individual)

    print("YAML files saved successfully!")
    print("Total consumed time:", time.time()-start_main)