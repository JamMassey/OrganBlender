import os
import pathlib

import cv2
import numpy as np
from scipy.ndimage import affine_transform
import SimpleITK as sitk
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from scipy.spatial.transform import Rotation


def image_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
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


def dicom_to_array(dicom, width=None, height=None, inter=cv2.INTER_LINEAR, voi_lut=True):
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


def dicom_to_array_og_aspratio(dicom, resize_to=1024, inter=cv2.INTER_LINEAR, voi_lut=True):
    data = dicom.pixel_array
    h, w = data.shape
    if w > h:
        data = dicom_to_array(dicom, width=resize_to, inter=inter, voi_lut=voi_lut)
    else:
        data = dicom_to_array(dicom, height=resize_to, inter=inter, voi_lut=voi_lut)
    return data


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_path_obj(path: str | pathlib.Path):
    return pathlib.Path(path) if not isinstance(path, pathlib.Path) else path


def process_directory(ipath, opath, width=None, height=None,
                      inter=cv2.INTER_LINEAR, voi_lut=True,
                      og_aspratio=False, resize_to=1024):
    ipath = ensure_path_obj(ipath)
    opath = ensure_path_obj(opath)
    create_directory(opath)
    # encoded_fn = os.fsencode(ipath)
    for img_fn in os.listdir(ipath):
        img_fn = pathlib.Path(img_fn)
        img_path = ensure_path_obj(f"{ipath}/{img_fn}")
        if img_path.suffix == ".dcm":
            dicom = dcmread(img_path)
            if og_aspratio:
                procd_arr = dicom_to_array_og_aspratio(dicom, resize_to=resize_to, inter=inter, voi_lut=voi_lut)
            else:
                procd_arr = dicom_to_array(dicom, width=width, height=height, inter=inter, voi_lut=voi_lut)
            cv2.imwrite(f"{opath}/{img_path.stem}.png", procd_arr)


def dicom_dir_to_array(dir_ipath, width=None, height=None, inter=cv2.INTER_LINEAR, voi_lut=True):
    dir_path = ensure_path_obj(dir_ipath)
    img_fns = os.listdir(dir_path)
    img_fns.sort()
    img_paths = [ensure_path_obj(f"{dir_ipath}/{img_fn}") for img_fn in img_fns]
    dcm_paths = list(filter(lambda x: x.suffix == ".dcm", img_paths))
    img_arrays = []
    for img_path in dcm_paths:
        img_array = dicom_to_array(dcmread(img_path), width=width, height=height, inter=inter, voi_lut=voi_lut)
        img_arrays.append(img_array)
    return np.stack(img_arrays, axis=-1)


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
    angle = np.radians(angle_degree)
    rotation = Rotation.from_euler('xyz', [angle] * 3, degrees=False)
    rotation_matrix = rotation.as_matrix()
    rotated_volume = affine_transform(volume, rotation_matrix)
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
    # Tri-linear interpolation for 3D images
    resampler.SetInterpolator(sitk.sitkLinear)
    rotated_volume = resampler.Execute(volume)
    # Extract the slice from the middle of the rotated volume
    slice_idx = rotated_volume.GetSize()[2] // 2
    extracted_slice = rotated_volume[:, :, slice_idx]
    return extracted_slice


def generate_mask(
        image_path: str,
        model_path: str = "../model/sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: str = "cpu",
        output_mode: str = "binary_mask"):
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
