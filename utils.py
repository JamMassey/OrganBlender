import os
import pathlib

import cv2
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import vtk
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def image_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def dicom_file_to_ary(dicom, width=None, height=None, inter=cv2.INTER_LINEAR, voi_lut=True):
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)

    data = (data - data.min()) / (data.max() - data.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data

    if width is not None or height is not None:
        data = image_resize(data, width, height, inter)

    data = (data * 255).astype(np.uint8)
    return data


def dicom_file_to_ary_original_aspect_ratio(dicom, resize_to=1024, inter=cv2.INTER_LINEAR, voi_lut=True):
    data = dicom.pixel_array
    h, w = data.shape
    if w > h:
        data = dicom_file_to_ary(dicom, width=resize_to, inter=inter, voi_lut=voi_lut)
    else:
        data = dicom_file_to_ary(dicom, height=resize_to, inter=inter, voi_lut=voi_lut)
    return data


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_directory(input_directory_path, output_directory_path, width=None, height=None, inter=cv2.INTER_LINEAR, voi_lut=True):
    # if string isnt a path object, convert it to one
    if not isinstance(input_directory_path, pathlib.Path):
        input_directory_path = pathlib.Path(input_directory_path)
    if not isinstance(output_directory_path, pathlib.Path):
        output_directory_path = pathlib.Path(output_directory_path)

    create_dir_if_not_exists(output_directory_path)
    input_directory_dir = os.fsencode(input_directory_path)
    for image_file_name in os.listdir(input_directory_dir):
        image_file_name = pathlib.Path(image_file_name)
        image_path = pathlib.Path(f"{input_directory_path}/{image_file_name}")
        if image_file_name.suffix == ".dcm":
            processed_ary = dicom_file_to_ary(dcmread(image_path), width=width, height=height, inter=inter, voi_lut=voi_lut)
            cv2.imwrite(f"{output_directory_path}/{image_file_name.stem}.png", processed_ary)


def process_directory_original_aspect_ratio(
    input_directory_path, output_directory_path, resize_to=1024, inter=cv2.INTER_LINEAR, voi_lut=True
):
    if not isinstance(input_directory_path, pathlib.Path):
        input_directory_path = pathlib.Path(input_directory_path)
    if not isinstance(output_directory_path, pathlib.Path):
        output_directory_path = pathlib.Path(output_directory_path)

    create_dir_if_not_exists(output_directory_path)
    for image_file_name in os.listdir(input_directory_path):
        image_file_name = pathlib.Path(image_file_name)
        image_path = pathlib.Path(f"{input_directory_path}/{image_file_name}")
        if image_file_name.suffix == ".dcm":
            processed_ary = dicom_file_to_ary_original_aspect_ratio(dcmread(image_path), resize_to=resize_to, inter=inter, voi_lut=voi_lut)
            cv2.imwrite(f"{output_directory_path}/{image_file_name.stem}.png", processed_ary)


def dicom_dir_to_3d_ary(input_directory_path, width=None, height=None, inter=cv2.INTER_LINEAR, voi_lut=True):
    if not isinstance(input_directory_path, pathlib.Path):
        input_directory_path = pathlib.Path(input_directory_path)

    image_file_names = os.listdir(input_directory_path)
    image_file_names.sort()
    image_file_names = [pathlib.Path(image_file_name) for image_file_name in image_file_names]
    image_paths = [pathlib.Path(f"{input_directory_path}/{image_file_name}") for image_file_name in image_file_names]
    image_paths = [image_path for image_path in image_paths if image_path.suffix == ".dcm"]

    image_arys = []
    for image_path in image_paths:
        image_ary = dicom_file_to_ary(dcmread(image_path), width=width, height=height, inter=inter, voi_lut=voi_lut)
        image_arys.append(image_ary)

    return np.stack(image_arys, axis=-1)


def acs_slice(volume, dim, slice_num):
    if dim == 0:
        return volume[slice_num, :, :]
    elif dim == 1:
        return volume[:, slice_num, :]
    elif dim == 2:
        return volume[:, :, slice_num]
    else:
        raise ValueError("dim must be 0, 1, or 2")


def extract_rotated_slice(volume, angle_degree):
    # Compute the rotation matrix. This is a simple example for rotation around one axis.
    # You might need a more complex transformation depending on the desired view.
    rotation_matrix = np.array(
        [
            [np.cos(np.radians(angle_degree)), -np.sin(np.radians(angle_degree)), 0],
            [np.sin(np.radians(angle_degree)), np.cos(np.radians(angle_degree)), 0],
            [0, 0, 1],
        ]
    )

    # Rotate the volume. Note: 'order=1' uses bilinear interpolation for 2D images.
    # For 3D volumes, this becomes trilinear interpolation.
    rotated_volume = scipy.ndimage.affine_transform(volume, rotation_matrix)

    # Extract the slice. Here we're extracting a slice in the middle of the rotated volume.
    # You can adjust as needed.
    slice_idx = rotated_volume.shape[2] // 2
    extracted_slice = rotated_volume[:, :, slice_idx]

    return extracted_slice


def extract_rotated_slice_with_sitk(volume_path, angle_x=None, angle_y=None, angle_z=None):
    # Read the volume
    volume = sitk.ReadImage(volume_path)

    # Set the rotation center
    center = volume.TransformContinuousIndexToPhysicalPoint([(sz - 1) / 2.0 for sz in volume.GetSize()])

    # Create a transformation
    transformation = sitk.Euler3DTransform()
    transformation.SetCenter(center)

    if angle_x is not None:
        transformation.SetRotation(np.radians(angle_x), 0, 0)
    if angle_y is not None:
        transformation.SetRotation(0, np.radians(angle_y), 0)
    if angle_z is not None:
        transformation.SetRotation(0, 0, np.radians(angle_z))

    # Resample the volume with the transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetTransform(transformation)
    resampler.SetInterpolator(sitk.sitkLinear)  # Trilinear interpolation for 3D images
    rotated_volume = resampler.Execute(volume)

    # Extract the slice from the middle of the rotated volume
    slice_idx = rotated_volume.GetSize()[2] // 2
    extracted_slice = rotated_volume[:, :, slice_idx]

    return extracted_slice


def extract_rotated_slice_with_sitk(volume_path, angle_x=None, angle_y=None, angle_z=None):
    # Read the volume
    volume = sitk.ReadImage(volume_path)

    # Set the rotation center
    center = volume.TransformContinuousIndexToPhysicalPoint([(sz - 1) / 2.0 for sz in volume.GetSize()])

    # Create a transformation
    transformation = sitk.Euler3DTransform()
    transformation.SetCenter(center)

    if angle_x is not None:
        transformation.SetRotation(np.radians(angle_x), 0, 0)
    if angle_y is not None:
        transformation.SetRotation(0, np.radians(angle_y), 0)
    if angle_z is not None:
        transformation.SetRotation(0, 0, np.radians(angle_z))

    # Resample the volume with the transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetTransform(transformation)
    resampler.SetInterpolator(sitk.sitkLinear)  # Trilinear interpolation for 3D images
    rotated_volume = resampler.Execute(volume)

    # Extract the slice from the middle of the rotated volume
    slice_idx = rotated_volume.GetSize()[2] // 2
    extracted_slice = rotated_volume[:, :, slice_idx]

    return extracted_slice


def generate_mask(
    image_path: str,
    model_path: str = "model/sam_vit_h_4b8939.pth",
    model_type: str = "vit_h",
    device: str = "cuda",
    output_mode: str = "binary_mask",
):
    """
    Generates masks for a given image using the Segment Anything model.

    Parameters:
    - image_path (str): Path to the input image.
    - model_path (str): Path to the Segment Anything model checkpoint.
    - model_type (str): Type of the model ('vit_l' by default).
    - device (str): Device to run the model on ('cuda' by default).

    Returns:
    - masks: Generated masks for the input image.
    """

    # Load the model
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)

    # Create the mask generator
    mask_generator = SamAutomaticMaskGenerator(model=sam, output_mode=output_mode)

    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate the masks
    masks = mask_generator.generate(image)

    return masks
