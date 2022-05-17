from typing import List
import pydicom
from pydicom import FileDataset
from src.slice.hu_thresholder import HounsfieldUnitThresholder
from src.slice.metadata import Metadata


class Slice:

    def __init__(self, image_path: str, metadata: Metadata) -> None:
        self._image_path: str = image_path
        self._image_index: int = int(self._image_path[-8:-4])
        self._dicom_file_dataset: FileDataset = read_dicom_file_dataset(
            self._image_path)
        self._metadata: Metadata = metadata
        self._metadata.change_dicom_file_dataset(self._dicom_file_dataset)
        self._z_value: float = (self._image_index -
                                1) * self._metadata.slice_thickness
        print('Image index', self._image_index)

    def extract_substance_3d_coordinates(self, lower_bound: int,
                                         upper_bound: int) -> List[List[float]]:
        hu_thresholder: HounsfieldUnitThresholder = HounsfieldUnitThresholder(
            lower_bound, upper_bound, self._z_value, self._metadata)

        return hu_thresholder.extract_substance_3d_coordinates()


def read_dicom_file_dataset(image_path: str) -> FileDataset:
    return pydicom.dcmread(image_path)
