import xml.etree.ElementTree as ET
import os
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# --- CONFIGURAÇÕES ---
# Substitua pelo caminho do seu dataset
DATA_DIR = "dataset"
# Defina suas classes aqui (deve incluir background como primeira classe)
CLASSES = ["__background__", "helmet", "vest", "gloves"]
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.005


# --- 1. DATASET CUSTOMIZADO ---
class EPIDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.anns = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, index):
        # Carregar imagem
        img_path = os.path.join(self.root, "images", self.imgs[index])
        img = Image.open(img_path).convert("RGB")

        # Carregar anotação (XML estilo Pascal VOC)
        ann_path = os.path.join(self.root, "annotations", self.anns[index])
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label_text = obj.find("name").text  # type: ignore
            if label_text in CLASSES:
                label_id = CLASSES.index(label_text)
                labels.append(label_id)

                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)  # type: ignore
                ymin = float(bndbox.find("ymin").text)  # type: ignore
                xmax = float(bndbox.find("xmax").text)  # type: ignore
                ymax = float(bndbox.find("ymax").text)  # type: ignore
                boxes.append([xmin, ymin, xmax, ymax])

        # Garantir tensores com shape correto mesmo quando não há objetos
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([index])

        # Calcular área apenas se houver boxes
        if boxes.shape[0] > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms: list = []
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))

    return torchvision.transforms.Compose(transforms)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Usando dispositivo: {device}")

    dataset = EPIDataset(DATA_DIR, get_transform(train=True))
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x))
    )

    num_classes = len(CLASSES)
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iter: {i}, Loss: {losses.item()}")
            i += 1

        lr_scheduler.step()
        print(f"Fim da época {epoch}")

    torch.save(model.state_dict(), "modelo_epi.pth")
    print("Modelo salvo em 'modelo_epi.pth'")


if __name__ == "__main__":
    main()
