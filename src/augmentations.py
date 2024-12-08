from tkinter import Image
from mltu.augmentors import RandomRotate
import numpy as np 
from mltu.annotations.images import Image 
from mltu.annotations.detections import Detections, Detection, BboxType
import typing


class RandomRotateFillWithMedian(RandomRotate): 
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

        # generate random border color 
        borderValue = np.median(image.numpy(), axis=(0, 1)).tolist()
        borderValue = [int(v) for v in borderValue]

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
