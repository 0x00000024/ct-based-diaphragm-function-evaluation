from src.respiration.base import Respiration


class Patient:

    def __init__(self, patient_id: str, image_dir_path: str) -> None:
        self._patient_id: str = patient_id
        self.inhalation: Respiration = Respiration(
            category='in',
            image_dir_path=image_dir_path,
            patient_id=self._patient_id)
        self.exhalation: Respiration = Respiration(
            category='ex',
            image_dir_path=image_dir_path,
            patient_id=self._patient_id)

    def get_inhalation(self) -> Respiration:
        return self.inhalation

    def get_exhalation(self) -> Respiration:
        return self.exhalation
