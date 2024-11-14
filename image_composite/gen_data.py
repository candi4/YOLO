import cv2
import yaml
import numpy as np


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
    rotated_img = cv2.warpAffine(translated_img, M, (int(2*r), int(2*r)), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    cropped_img = crop_nonzero(rotated_img)
    return cropped_img


def resize(img, scale):

    height, width = img.shape[:2]

    new_width = width * scale
    new_height = height * scale

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
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



def generate_yolo_label(yaml_path, img_width, img_height, x_center, y_center, obj_width, obj_height, class_id=0):
    # Calculate YOLO annotation format values
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = obj_width / img_width
    height_norm = obj_height / img_height

    # YAML data
    data = {
        'path': yaml_path,
        'x_center': x_center_norm,
        'y_center': y_center_norm,
        'width': width_norm,
        'height': height_norm,
        'class': class_id
    }
    return data

if __name__ == "__main__":
    # Load background and object images
    background = cv2.imread(r'C:\Users\82105\code_temp\YOLO\image_composite\background\ddca3f92-4b8e-4672-bb6b-f3594ad4e304.jpg', cv2.IMREAD_GRAYSCALE)
    object_img = cv2.imread(r'C:\Users\82105\code_temp\YOLO\image_composite\object\(ry0)(rx0)(delx0.1).png', cv2.IMREAD_GRAYSCALE)


    x=600
    y=1200
    bg_scale=5
    obj_scale=1
    combined_img, background, object_img = overlay_image(background=background, bg_scale=bg_scale, object_img=object_img, obj_scale=obj_scale, x=x, y=y, angle=30,
                                bg_weight=0.6, obj_weight=0.8, gamma=0)

    # Save combined image
    cv2.imwrite("combined_image.png", combined_img)

    # YAML Annotation generation
    yaml_data = generate_yolo_label("combined_image.png", combined_img.shape[1], combined_img.shape[0],
                                        x + (object_img.shape[1] // 2), y + (object_img.shape[0] // 2),
                                        object_img.shape[1], object_img.shape[0])

    # Save YAML
    with open("yolo_annotation.yaml", "w") as file:
        yaml.dump(yaml_data, file)

    print("YAML file saved successfully!")
