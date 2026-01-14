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

    EPI_CLASSES = ["__background__", "helmet", "vest", "gloves"]

    def __init__(self, epi_model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inicializando detector no dispositivo: {self.device}")

        # Modelo COCO para detectar pessoas
        print("  - Carregando modelo COCO para detecção de pessoas...")
        self.person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        self.person_model = self.person_model.to(self.device).eval()

        # Modelo customizado para EPIs
        self.epi_model = None
        if epi_model_path:
            print(f"  - Carregando modelo EPI de: {epi_model_path}")
            self.epi_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights="DEFAULT"
            )
            in_features = self.epi_model.roi_heads.box_predictor.cls_score.in_features
            self.epi_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

            state_dict = torch.load(
                epi_model_path, map_location=self.device, weights_only=True
            )
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
            epi_predictions = (
                self.epi_model(img_tensor)[0] if self.epi_model else None
            )

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
                        box = epi_predictions["boxes"][i].cpu().numpy().astype(int)
                        label = self.EPI_CLASSES[label_id]
                        combined.append((box, label, score))

        return combined

    def _calculate_iou(self, box1, box2) -> float:
        """Calcula IoU entre duas caixas."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _check_epi_usage(
        self, results: list[tuple], iou_threshold: float = 0.1
    ) -> list[dict]:
        """Verifica uso de EPIs por pessoa."""
        persons = [
            (box, idx)
            for idx, (box, label, _) in enumerate(results)
            if label == "person"
        ]
        epis = {
            "helmet": [
                (box, idx)
                for idx, (box, label, _) in enumerate(results)
                if label == "helmet"
            ],
            "vest": [
                (box, idx)
                for idx, (box, label, _) in enumerate(results)
                if label == "vest"
            ],
            "gloves": [
                (box, idx)
                for idx, (box, label, _) in enumerate(results)
                if label == "gloves"
            ],
        }

        person_epi_status = []

        for person_box, person_idx in persons:
            status = {
                "person_idx": person_idx,
                "box": person_box.tolist(),
                "helmet": False,
                "vest": False,
                "gloves": False,
            }

            for epi_type, epi_list in epis.items():
                for epi_box, _ in epi_list:
                    iou = self._calculate_iou(person_box, epi_box)
                    if iou > iou_threshold:
                        status[epi_type] = True
                        break

            person_epi_status.append(status)

        return person_epi_status
