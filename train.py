import os
import random
import xml.etree.ElementTree as ET
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

# --- CONFIGURAÇÕES ---
DATA_DIR = "dataset"
CLASSES = ["__background__", "helmet", "vest", "gloves"]
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.005
TRAIN_SPLIT = 0.8  # 80% treino, 20% validação


# --- UTILS DE TRANSFORMAÇÃO (Corrige o bug de flip) ---
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            # Flip das caixas: xmin = width - old_xmax, xmax = width - old_xmin
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ColorJitter:
    """Aplica distorções de cor apenas na imagem."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.transform = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        # ColorJitter espera PIL ou Tensor. Como aplicamos antes de ToTensor, é PIL.
        image = self.transform(image)
        return image, target


def get_transform(train):
    transforms = []
    if train:
        # Adiciona variação de cor para robustez (ajuda a diferenciar cabelo de capacete)
        transforms.append(ColorJitter())
        transforms.append(ToTensor())
        transforms.append(RandomHorizontalFlip(0.5))
    else:
        transforms.append(ToTensor())
    return Compose(transforms)


# --- DATASET ---
class EPIDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.anns = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "images", self.imgs[index])
        ann_path = os.path.join(self.root, "annotations", self.anns[index])

        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(ann_path)
        root_xml = tree.getroot()

        boxes = []
        labels = []

        for obj in root_xml.findall("object"):
            label_text = obj.find("name").text
            if label_text in CLASSES:
                label_id = CLASSES.index(label_text)
                labels.append(label_id)

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    # Necessário para evitar erro de pickle ao usar SubsetRandomSampler com transforms diferentes
    def set_transforms(self, transforms):
        self.transforms = transforms


def collate_fn(batch):
    return tuple(zip(*batch))


# --- MODELO ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --- LOOP DE TREINO ---
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Usando dispositivo: {device}")

    try:
        # Preparar Dataset
        dataset = EPIDataset(DATA_DIR, None)  # Transforms serão aplicados no Loop ou via Wrapper

        if len(dataset) == 0:
            print("ERRO: Dataset vazio ou não encontrado.")
            return

        # Split de índices
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(TRAIN_SPLIT * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]

        print(f"Tamanho do Dataset: {dataset_size}")
        print(f"Treino: {len(train_indices)}, Validação: {len(val_indices)}")

        # Wrappers customizados para aplicar transforms corretos em cada split
        # Como o SubsetRandomSampler apenas seleciona índices, precisamos de uma forma de injetar transforms.
        # A maneira mais limpa sem criar duas instâncias de dataset carregando tudo 2x é ter subclasses ou wrappers.
        # Simplificação: Criar dois datasets apontando para o mesmo lugar.
        train_dataset = EPIDataset(DATA_DIR, get_transform(train=True))
        val_dataset = EPIDataset(DATA_DIR, get_transform(train=False))

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices), num_workers=2, collate_fn=collate_fn
        )
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_indices), num_workers=2, collate_fn=collate_fn)

        num_classes = len(CLASSES)
        model = get_model(num_classes)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_val_loss = float("inf")

        for epoch in range(NUM_EPOCHS):
            # --- TREINAMENTO ---
            model.train()
            train_loss_epoch = 0
            train_batches = 0

            for images, targets in train_loader:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss_epoch += losses.item()
                train_batches += 1

            avg_train_loss = train_loss_epoch / train_batches if train_batches > 0 else 0

            # --- VALIDAÇÃO ---
            # Faster R-CNN retorna losses durante training=True, e predições durante training=False.
            # Para calcular loss de validação, precisamos colocar em modo train() mas sem backprop.
            # Existe um truque: manter train() mas usar torch.no_grad().

            model.train()  # Mantém modo treino para retornar losses
            val_loss_epoch = 0
            val_batches = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    losses = sum(loss_dict.values())
                    val_loss_epoch += losses.item()
                    val_batches += 1

            avg_val_loss = val_loss_epoch / val_batches if val_batches > 0 else 0

            lr_scheduler.step()

            print(f"Epoch: {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Salvar melhor modelo
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "modelo_epi.pth")
                print(f"  -> Modelo salvo (Melhor Val Loss: {best_val_loss:.4f})")

        print("Treinamento finalizado.")
    except Exception as e:
        print(f"Erro fatal: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
