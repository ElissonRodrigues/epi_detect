"""
Módulo de detecção de EPIs reutilizando a lógica do main.py.
Otimizado para uso em streaming web.
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import numpy as np

CONFIDENCE_THRESHOLD = 0.7


class EPIDetector:
    """Detector de EPIs usando Faster R-CNN."""

    EPI_CLASSES = ["__background__", "helmet", "vest", "gloves", "_head_", "_not_helmet_"]
    # Classes que são ignoradas na exibição (usadas apenas para treinamento)
    IGNORED_CLASSES = ["_head_", "_not_helmet_"]

    def __init__(self, epi_model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
        print(f"Inicializando detector no dispositivo: {self.device}")

        # Modelo COCO para detectar pessoas
        print("  - Carregando modelo COCO para detecção de pessoas...")
        self.person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self.person_model = self.person_model.to(self.device).eval()

        # Modelo customizado para EPIs
        self.epi_model = None
        if epi_model_path:
            print(f"  - Carregando modelo EPI de: {epi_model_path}")
            self.epi_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
            in_features = self.epi_model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
            self.epi_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.EPI_CLASSES))

            state_dict = torch.load(epi_model_path, map_location=self.device, weights_only=True)
            self.epi_model.load_state_dict(state_dict)
            self.epi_model = self.epi_model.to(self.device).eval()

        print("Detector inicializado com sucesso!")

    def detect(self, frame: np.ndarray) -> tuple[list[tuple], list[dict]]:
        """
        Detecta pessoas e EPIs no frame.

        Args:
            frame: Imagem BGR do OpenCV

        Returns:
            Tuple com:
            - Lista de detecções (box, label, score)
            - Lista de status de EPI por pessoa
        """
        # Converter para tensor
        img_tensor = F.to_tensor(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            person_predictions = self.person_model(img_tensor)[0]
            epi_predictions = self.epi_model(img_tensor)[0] if self.epi_model else None

        # Combinar detecções
        results = self._combine_predictions(person_predictions, epi_predictions)

        # Verificar uso de EPIs
        epi_status = self._check_epi_usage(results)

        return results, epi_status

    def _combine_predictions(self, person_predictions, epi_predictions) -> list[tuple]:
        """Combina predições dos dois modelos."""
        combined = []

        # Pessoas do modelo COCO
        if person_predictions:
            for i, score in enumerate(person_predictions["scores"].tolist()):
                if score > CONFIDENCE_THRESHOLD:
                    label_id = person_predictions["labels"][i].item()
                    if label_id == 1:  # person no COCO
                        box = person_predictions["boxes"][i].cpu().numpy().astype(int)
                        combined.append((box, "person", score))

        # EPIs do modelo customizado
        if epi_predictions:
            for i, score in enumerate(epi_predictions["scores"].tolist()):
                if score > CONFIDENCE_THRESHOLD:
                    label_id = epi_predictions["labels"][i].item()
                    if 0 < label_id < len(self.EPI_CLASSES):
                        label = self.EPI_CLASSES[label_id]
                        # Ignora classes de referência negativa
                        if label in self.IGNORED_CLASSES:
                            continue
                        box = epi_predictions["boxes"][i].cpu().numpy().astype(int)
                        combined.append((box, label, score))

        return combined

    def _calculate_overlap(self, person_box, epi_box) -> float:
        """
        Calcula a sobreposição do EPI (box2) em relação à pessoa (box1).
        Usa Intersection over EPI Area para lidar com a diferença de escala.
        """
        x1_p, y1_p, x2_p, y2_p = person_box
        x1_e, y1_e, x2_e, y2_e = epi_box

        x1_inter = max(x1_p, x1_e)
        y1_inter = max(y1_p, y1_e)
        x2_inter = min(x2_p, x2_e)
        y2_inter = min(y2_p, y2_e)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        epi_area = (x2_e - x1_e) * (y2_e - y1_e)

        return inter_area / epi_area if epi_area > 0 else 0.0

    def _validate_anatomical_position(self, person_box, epi_box, epi_type) -> bool:
        """
        Verifica se o EPI está em uma posição anatômica plausível.
        Ex: Capacete deve estar no topo, colete no meio.
        """
        _, y1_p, _, y2_p = person_box
        _, y1_e, _, y2_e = epi_box

        person_height = y2_p - y1_p
        if person_height == 0:
            return False

        epi_center_y = (y1_e + y2_e) / 2
        # Posição relativa do centro do EPI (0.0 = topo da pessoa, 1.0 = pé da pessoa)
        relative_pos = (epi_center_y - y1_p) / person_height

        if epi_type == "helmet":
            # Capacete deve estar na parte superior (topo 40%)
            # Evita detectar capacete segurado no peito ou cintura
            return relative_pos < 0.40

        if epi_type == "vest":
            # Colete deve estar no tronco (entre 10% e 80%)
            # Evita detectar colete jogado no chão perto do pé ou segurado muito alto
            return 0.10 < relative_pos < 0.80

        # Luvas podem estar em qualquer lugar (mãos se movem muito)
        return True

    def _check_epi_usage(self, results: list[tuple], threshold: float = 0.1) -> list[dict]:
        """Verifica uso de EPIs por pessoa."""
        persons = [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "person"]
        epis = {
            "helmet": [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "helmet"],
            "vest": [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "vest"],
            "gloves": [(box, idx) for idx, (box, label, _) in enumerate(results) if label == "gloves"],
        }

        person_epi_status = []

        for person_number, (person_box, person_idx) in enumerate(persons, start=1):
            status = {
                "person_idx": person_idx,
                "person_number": person_number,  # Número visível para correlação com o vídeo
                "box": person_box.tolist(),
                "helmet": False,
                "vest": False,
                "gloves": False,
            }

            for epi_type, epi_list in epis.items():
                for epi_box, _ in epi_list:
                    overlap = self._calculate_overlap(person_box, epi_box)
                    if overlap > threshold:
                        # Verifica se faz sentido anatomicamente
                        if self._validate_anatomical_position(person_box, epi_box, epi_type):
                            status[epi_type] = True
                            break

            person_epi_status.append(status)

        return person_epi_status
