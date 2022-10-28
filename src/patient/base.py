from colorama import Fore
from src.respiration.base import Respiration


class Patient:

    def __init__(self, patient_id: str, image_dir_path: str) -> None:
        self._patient_id: str = patient_id
        self._image_dir_path: str = image_dir_path
        self._analyze_respiration(category='in')
        self._analyze_respiration(category='ex')

    def _init_respiration(self, category: str, roi: str) -> Respiration:
        return Respiration(patient_id=self._patient_id,
                           category=category,
                           image_dir_path=self._image_dir_path,
                           roi=roi)

    # def _analyze_respiration(self, category: str) -> None:
    #     print(Fore.BLUE + f'Analyzing {self._patient_id} {category}halation')
    #     roi: str = 'lung'
    #     while roi != '0':
    #         r: Respiration = self._init_respiration(category=category, roi=roi)
    #         r.calculate_lung_points_coordinates()
    #         roi = r.extract_lung_base()
    #         if roi != '0':
    #             continue
    #         r.synthesize_surface_mesh()
    #
    #     print(Fore.BLUE + f'Analyzing {self._patient_id} {category}halation done')

    def _analyze_respiration(self, category: str) -> None:
        print(Fore.BLUE + f'Analyzing {self._patient_id} {category}halation')
        roi: str = 'thorax'
        r: Respiration = self._init_respiration(category=category, roi=roi)
        # r.calculate_lung_points_coordinates()
        r.extract_lung_base()
        r.synthesize_surface_mesh()

        print(Fore.BLUE + f'Analyzing {self._patient_id} {category}halation done')
