import cv2 as cv
from nptyping import NDArray, Shape, Int8, Int16
from pydicom import FileDataset
from src.slice.denoiser import Denoiser


# Only create one instance for Metadata class to reduce repeated reading of public metadata
class Metadata:

    def __init__(self, dicom_file_dataset: FileDataset) -> None:
        # Load common data for dicom images
        self._dicom_file_dataset: FileDataset = dicom_file_dataset
        self.rows: int = int(self._dicom_file_dataset[0x28, 0x10].value)
        self.columns: int = int(self._dicom_file_dataset[0x28, 0x11].value)
        self.row_spacing: float = float(self._dicom_file_dataset[0x28,
                                                                 0x30].value[0])
        self.col_spacing: float = float(self._dicom_file_dataset[0x28,
                                                                 0x30].value[1])
        self.rescale_intercept: int = int(
            self._dicom_file_dataset[0x28, 0x1052].value)
        self.rescale_slope: int = int(self._dicom_file_dataset[0x28,
                                                               0x1053].value)
        rows = self.rows
        columns = self.columns
        self.pixel_data: NDArray[Shape[rows, columns], Int16] | None = None
        self.denoised_data: NDArray[Shape[rows, columns], Int8] | None = None
        self.slice_thickness: float | None = None

    def change_dicom_file_dataset(self,
                                  dicom_file_dataset: FileDataset) -> None:
        self._dicom_file_dataset = dicom_file_dataset
        self.load_special_data()

    def load_special_data(self) -> None:
        self.pixel_data = self._dicom_file_dataset.pixel_array
        self.denoised_data = cv.convertScaleAbs(self.pixel_data)
        denoiser: Denoiser = Denoiser(self.rows, self.columns,
                                      self.denoised_data)
        self.denoised_data = denoiser.set_lungs_as_roi()
        self.slice_thickness = float(self._dicom_file_dataset[0x18, 0x50].value)
