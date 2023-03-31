from os import listdir
from os.path import join
from typing import List
import pydicom
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir


class Metadata:

    def __init__(self, category: str, image_path: str) -> None:
        self.category: str = category
        self.dicom_file_dataset: FileDataset | DicomDir = pydicom.dcmread(
            self.image_path)
        self.rows: int = int(self.dicom_file_dataset[0x28, 0x10].value)
        self.columns: int = int(self.dicom_file_dataset[0x28, 0x11].value)
        self.slice_thickness: float = float(self.dicom_file_dataset[0x18,
                                                                    0x50].value)
        self.row_spacing: float = float(
            self.dicom_file_dataset[0x28, 0x30].value[0]) * 10**-1
        self.col_spacing: float = float(
            self.dicom_file_dataset[0x28, 0x30].value[1]) * 10**-1
        self.rescale_intercept: int = int(self.dicom_file_dataset[0x28,
                                                                  0x1052].value)
        self.rescale_slope: int = int(self.dicom_file_dataset[0x28,
                                                              0x1053].value)
        print('Called metadata class')