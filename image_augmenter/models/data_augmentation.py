import os
import numpy as np
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import json
from pathlib import Path

from image_augmenter.models.numpy_encoder import NumpyEncoder


class PolyAugmenter:
    """
    Cls for augmenting training dataset.
    ---

    Args:
        AUGMENTERS - Choose augmenters to be used for expanding dataset.

        via_annotations - original annotations
        img_info - dict containing filename and np.ndarray of the image {name: img(np.array)}
        validate - Is true creates an extra set of images that shows all polygons
        POI - list of all PolygonsOnImages, hold the adjusted polygon information
        augmented_via - new dict in original format that holds all new polygon values as well as new file names
        augmented_images - Contains all images as well as their new names.

    """

    AUGMENTERS = [
        iaa.Fliplr(0, name='Original'),  # Keeps original image
        # iaa.Rot90(1, keep_size=False, name='rot_90'),  # Rotate by 90 degrees
        # iaa.Rot90(2, keep_size=False, name='rot_180'),  # Rotate by 180 degrees
        # iaa.Rot90(3, keep_size=False, name='rot_270'),  # Rotate by 270 degrees
        #
        # # Mirror, rotate by 90 degrees and darken image
        # iaa.Sequential(
        #     [iaa.Fliplr(1),
        #      iaa.Rot90(1, keep_size=False),
        #      iaa.Multiply(0.5)], name='lr_rot_90_dark'),
        # # Mirror, rotate by 180 degrees and brighten image
        # iaa.Sequential(
        #     [iaa.Fliplr(1),
        #      iaa.Rot90(2, keep_size=False),
        #      iaa.Multiply(1.5)], name='lr_rot_180_bright'),
        # # Mirror, rotate by 270 degrees and randomly darken/brighten image
        # iaa.Sequential(
        #     [iaa.Fliplr(1),
        #      iaa.Rot90(3, keep_size=False),
        #      iaa.Multiply((0.5, 1.5))], name='lr_rot_270_random'),
    ]

    def __init__(self, annotations_path: str, images_path: str, validate: bool = False):
        self.via_annotations: dict = json.load(open(annotations_path))
        self.img_info: dict = {file: cv2.imread(os.path.join(images_path, file)) for file in os.listdir(images_path) if file.endswith('PNG')}
        self.validate: bool = validate
        self.POI: list = []
        self.augmented_via: dict = {}
        self.augmented_images: dict = {}

        if self.validate:  # Creates an extra set of images for validation purposes only, will take longer to run
            self.overlaid_images: dict = {}

    @staticmethod
    def get_centre(values: list, shape: int) -> int:
        """Calculates normalized x_y centre values"""
        return ((min(values) + max(values)) / 2) / shape

    @staticmethod
    def get_size(values: list, shape: int) -> int:
        """Calculates normalized x_y centre values"""
        return (max(values) - min(values)) / shape

    def transform_yolo(self, name_dict: dict, file_name: str, name: str, shape: str, all_points_x: list, all_points_y: list) -> str:
        """
        Transforms annotations to normalized xywh format.

        i.e, class_name x_centre y_centre width height
        """
        yolo_name = name_dict[name]

        X_centre = self.get_centre(all_points_x, self.augmented_images[file_name].shape[1])
        Y_centre = self.get_centre(all_points_y, self.augmented_images[file_name].shape[0])
        width = self.get_size(all_points_x, self.augmented_images[file_name].shape[1])
        height = self.get_size(all_points_y, self.augmented_images[file_name].shape[0])

        return f'{yolo_name} {X_centre} {Y_centre} {width} {height}'

    def save_as_yolo_annotation(self, save_directory, **name_dict) -> None:
        """Reads classnames from name_dict and creates annotations in YOLO *.txt format"""
        for file_name, val in self.augmented_via.items():
            yolo_annotations = [self.transform_yolo(name_dict, file_name, **v) for v in val]

            with open(f'{save_directory}/{file_name[:-4]}.txt', 'w') as file:
                file.write('\n'.join(yolo_annotations))

    def _save_validations(self, save_directory: str) -> None:
        """Saves images with annotations on them"""
        [cv2.imwrite(f'{save_directory}/{name}', image) for name, image in self.overlaid_images.items()]

    def _save_annotations(self, save_directory: str) -> None:
        """Saves annotations back to original format"""
        with open(f'{save_directory}/main_dict_augmented.json', 'w') as f:
            json.dump(self.augmented_via, f, cls=NumpyEncoder)

    def _save_images(self, save_directory: str) -> None:
        """Saves images with new names according to augmentation"""
        [cv2.imwrite(f'{save_directory}/{name}', image) for name, image in self.augmented_images.items()]

    def save_all(self, save_directory: str) -> None:
        """Saves all annotations and images."""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self._save_images(save_directory)
        self._save_annotations(save_directory)

        if self.validate:
            self._save_validations(save_directory)

    def extract_and_transform(self, name: str) -> list:
        """
        1. Extracts all annotations from original VIA annotator file
        2. Transforms the annotations from x = [1, 2, 3], y = [1, 2, 3] -> (1, 1), (2, 2), (3, 3)
        """

        polys = [r['shape_attributes'] for k, a in self.via_annotations.items() for r in a['regions'] if name in k]
        return [[(x, y) for x, y in zip(shape['all_points_x'], shape['all_points_y'])] for shape in polys]

    def make_polys(self) -> None:
        """Reads all annotations, appends them as PolygonsOnImage object to self.POI."""

        for name, np_img in self.img_info.items():
            points_x_y = self.extract_and_transform(name)
            poly_objects = [Polygon(points, label='polygon') for points in points_x_y]  # Creates polygon objects
            polys_oi = PolygonsOnImage(poly_objects, shape=np_img.shape)  # Creates PolygonOnImage object

            self.POI.append(polys_oi)

    def invoke_augmenters(self) -> dict:
        """
        Loop through and call for each augmenter for each image specified in cls.AUGMENTERS

        save as {
        IMG_NAME: {
            AUG_NAME: [<class 'numpy.ndarray'>, <class 'imgaug.augmentables.polys.PolygonsOnImage'>]
            }
        }
        """
        augmented_images = {}
        for image, POI in zip(self.img_info.items(), self.POI):
            augmented_images[image[0]] = {augmenter.name: augmenter(image=image[1], polygons=POI) for augmenter in self.AUGMENTERS}

        return augmented_images

    def recreate_dict(self, img_name: str, augmentations: dict) -> None:
        """Build dict in format of original annotations with new filenames"""

        for aug_name, values in augmentations.items():  # {AUG_NAME: [<class 'numpy.ndarray'>, <class# 'imgaug.augmentables.polys.PolygonsOnImage'>]}
            image, polygon = values
            file_name = f'{aug_name}{img_name}'

            # Build new dict -> {file_name: [{name, shape, all_points_x, all_points_y}, {name, shape, all_points_x, all_points_y}, ...]}
            self.augmented_via[file_name] = [{
                'name': 'snag',
                'shape': poly.label,
                'all_points_x': np.clip(poly.xx, 0, image.shape[1] - 1),  # Set max val to width - 1
                'all_points_y': np.clip(poly.yy, 0, image.shape[0] - 1)  # Set max val to height - 1
            }
                for poly in polygon]

            # Save images with their new file_name: {name: np.array(image)}
            self.augmented_images[file_name] = image

            if self.validate:  # Draws all polygons on image
                self.overlaid_images[f'val_{file_name}'] = polygon.draw_on_image(image)

    def augment_images(self) -> None:

        if not self.POI:  # Do not append more than one set of POIs to self.POI
            self.make_polys()

        augmented = self.invoke_augmenters()

        for img_name, augmentations in augmented.items():
            self.recreate_dict(img_name, augmentations)
