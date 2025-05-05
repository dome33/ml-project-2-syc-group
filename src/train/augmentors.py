import cv2
import typing
import numpy as np

import logging

from mltu.annotations.detections import Detections
from mltu.augmentors import randomness_decorator
from mltu.augmentors import Augmentor

from mltu.annotations.images import Image



class RandomBrightness(Augmentor):
    """ Randomly adjust image brightness """

    def __init__(
            self,
            random_chance: float = 0.5,
            delta: int = 100,
            log_level: int = logging.INFO,
            augment_annotation: bool = False
    ) -> None:
        """ Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool, optional): If True, the annotation will be adjusted as well. Defaults to False.
        """
        super(RandomBrightness, self).__init__(random_chance, log_level, augment_annotation)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    def augment(self, image: Image, value: float) -> Image:
        """ Augment image brightness """

        img = image.numpy()  # get NumPy array

        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Grayscale
            bright = img.astype(np.float32) * value
            new_image = np.clip(bright, 0, 255).astype(np.uint8)
        else:
            hsv = np.array(image.HSV(), dtype=np.float32)

            hsv[:, :, 1] = hsv[:, :, 1] * value
            hsv[:, :, 2] = hsv[:, :, 2] * value

            hsv = np.uint8(np.clip(hsv, 0, 255))

            new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image.update(new_image)

        return image

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly adjust image brightness

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        value = 1 + np.random.uniform(-self._delta, self._delta) / 255

        image = self.augment(image, value)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation, value)

        return image, annotation


class RandomRotate(Augmentor):
    """ Randomly rotate image"""

    def __init__(
            self,
            random_chance: float = 0.5,
            angle: typing.Union[int, typing.List] = 30,
            borderValue: typing.Tuple[int, int, int] = None,
            log_level: int = logging.INFO,
            augment_annotation: bool = True
    ) -> None:
        """ Randomly rotate image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            angle (int, list): Integer value or list of integer values for image rotation
            borderValue (tuple): Tuple of 3 integers, setting border color for image rotation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): If True, the annotation will be adjusted as well. Defaults to True.
        """
        super(RandomRotate, self).__init__(random_chance, log_level, augment_annotation)

        self._angle = angle
        self._borderValue = borderValue

    @staticmethod
    def rotate_image(image: np.ndarray, angle: typing.Union[float, int], borderValue: tuple = (255,),
                     return_rotation_matrix: bool = False) -> np.ndarray:
        # grab the dimensions of the image and then determine the centre
        height, width = image.shape[:2]
        center_x, center_y = (width // 2, height // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        # Handle grayscale vs color
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            img = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue[0])
        else:
            img = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)

        if return_rotation_matrix:
            return img, M

        return img

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly rotate image

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation
        """
        # check if angle is list of angles or a single angle value
        if isinstance(self._angle, list):
            angle = float(np.random.choice(self._angle))
        else:
            angle = float(np.random.uniform(-self._angle, self._angle))

        img_array = image.numpy()
        is_gray = len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1)

        if self._borderValue is None:
            borderValue = (255,) if is_gray else tuple(np.random.randint(0, 255, 3))
        else:
            borderValue = (self._borderValue[0],) if is_gray else self._borderValue

        
        try:

            img, rotMat = self.rotate_image(image.numpy(), angle, borderValue, return_rotation_matrix=True)

            if self._augment_annotation:
                if isinstance(annotation, Image):
                    # perform the actual rotation and return the annotation image
                    annotation_image = self.rotate_image(annotation.numpy(), angle, borderValue=(0, 0, 0))
                    annotation.update(annotation_image)
                elif isinstance(annotation, Detections):
                    height, width = img.shape[:2]
                    for detection in annotation:
                        detection.dot(rotMat, width, height)

            image.update(img)

            return image, annotation
        except:
            print("Exception in rotation image, returning original")
            return image, annotation


class RandomHorizontalShear(Augmentor):
    """ Randomly apply horizontal shear to an image. """

    def __init__(
        self,
        random_chance: float = 0.5,
        max_shear_factor: float = 0.3,
        borderValue: typing.Union[int, typing.Tuple[int, int, int]] = 255,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        """
        Args:
            random_chance (float): Probability of applying the shear. Default is 0.5.
            max_shear_factor (float): Maximum shear factor in either direction (positive or negative). Default is 0.3.
            borderValue (int or tuple): Border color to use for padding after transform. Default is 255 (white).
            augment_annotation (bool): If True, apply same shear to annotation (e.g., masks, detections). Default is False.
        """
        super().__init__(random_chance, log_level, augment_annotation)
        self.max_shear_factor = max_shear_factor
        self.borderValue = borderValue

    def shear_image(self, img: np.ndarray, shear_factor: float, border_value) -> np.ndarray:
        h, w = img.shape[:2]

        M = np.array([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        nW = int(w + abs(shear_factor * h))

        if len(img.shape) == 2 or (img.shape[2] == 1):  # grayscale
            border_value = border_value if isinstance(border_value, int) else border_value[0]
        else:
            border_value = border_value if isinstance(border_value, tuple) else (border_value,) * 3

        return cv2.warpAffine(img, M, (nW, h), borderValue=border_value)

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        shear_factor = np.random.uniform(-self.max_shear_factor, self.max_shear_factor)

        img_array = image.numpy()
        is_gray = len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1)

        border_value = (255,) if is_gray else (255, 255, 255)
        if isinstance(self.borderValue, int) or isinstance(self.borderValue, tuple):
            border_value = self.borderValue

        try:

            sheared_img = self.shear_image(img_array, shear_factor, border_value)

            if self._augment_annotation and isinstance(annotation, Image):
                sheared_annotation = self.shear_image(annotation.numpy(), shear_factor, 0)
                annotation.update(sheared_annotation)

            image.update(sheared_img)
            return image, annotation
        
        except:
            print("Exception in shear image, returning original")
            return image, annotation


class RandomHorizontalScale(Augmentor):
    """Randomly scale image horizontally by a small factor (±10%)."""

    def __init__(
        self,
        random_chance: float = 0.5,
        scale_range: typing.Tuple[float, float] = (0.8, 1.2),
        borderValue: typing.Union[int, typing.Tuple[int, int, int]] = 255,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        """
        Args:
            random_chance (float): Probability of applying the transform. Default is 0.5.
            scale_range (tuple): Min and max scale factor for width. E.g. (0.9, 1.1).
            borderValue (int or tuple): Border color. Default is 255 (white).
            log_level (int): Logging level.
            augment_annotation (bool): Whether to apply to annotation too.
        """
        super().__init__(random_chance, log_level, augment_annotation)
        self.scale_range = scale_range
        self.borderValue = borderValue

    def scale_image(self, img: np.ndarray, scale_factor: float, border_value) -> np.ndarray:
        h, w = img.shape[:2]
        new_w = int(w * scale_factor)

        scaled_img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)

        # Resize canvas back to original width with padding or cropping
        if new_w < w:
            pad = (w - new_w) // 2
            if len(img.shape) == 2 or img.shape[2] == 1:
                canvas = np.full((h, w), border_value, dtype=np.uint8)
            else:
                canvas = np.full((h, w, 3), border_value, dtype=np.uint8)
            canvas[:, pad:pad + new_w] = scaled_img
            return canvas
        elif new_w > w:
            crop = (new_w - w) // 2
            return scaled_img[:, crop:crop + w]
        else:
            return scaled_img

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])

        img_array = image.numpy()
        is_gray = len(img_array.shape) == 2 or (img_array.ndim == 3 and img_array.shape[2] == 1)

        border_value = self.borderValue if is_gray or isinstance(self.borderValue, int) else self.borderValue

        try:
            scaled_img = self.scale_image(img_array, scale_factor, border_value)

            image.update(scaled_img)

            if self._augment_annotation and isinstance(annotation, Image):
                scaled_ann = self.scale_image(annotation.numpy(), scale_factor, 0)
                annotation.update(scaled_ann)

            return image, annotation
        except:
            print("Exception in scale image, returning original")
            return image, annotation

        