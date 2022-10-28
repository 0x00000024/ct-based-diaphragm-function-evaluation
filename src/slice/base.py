from typing import List
import pydicom
from pydicom import FileDataset
from nptyping import NDArray, Shape, Int8
from colorama import Fore
import cv2 as cv
from src.slice.hu_thresholder import HounsfieldUnitThresholder
from src.slice.metadata import Metadata
from src.slice.denoiser import Denoiser


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
        print(Fore.CYAN, f'Analyzing image index: {self._image_index}')

    def extract_substance_3d_coordinates(self, lower_bound: int,
                                         upper_bound: int,
                                         roi: str) -> List[List[float]]:
        denoiser: Denoiser = Denoiser(
            self._metadata.rows, self._metadata.columns,
            cv.convertScaleAbs(self._metadata.pixel_data))
        denoised_data: NDArray[Shape["*, *"], Int8] | None = None
        if roi == 'thorax':
            denoised_data: NDArray[Shape["*, *"],
                                   Int8] = denoiser.set_thorax_as_roi()
        if roi == 'lung':
            denoised_data: NDArray[Shape["*, *"],
                                   Int8] = denoiser.set_lung_as_roi()

        hu_thresholder: HounsfieldUnitThresholder = HounsfieldUnitThresholder(
            lower_bound, upper_bound, self._z_value, self._metadata,
            denoised_data)

        return hu_thresholder.extract_substance_3d_coordinates()


def read_dicom_file_dataset(image_path: str) -> FileDataset:
    return pydicom.dcmread(image_path)
