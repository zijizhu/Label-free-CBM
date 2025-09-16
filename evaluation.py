import argparse
import logging
import re
import sys
from logging import Logger
from pathlib import Path

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy

from eval.concept_trustworthiness import Cub2011Eval, evaluate_concept_trustworthiness
from eval.local_parts import id_to_attributes
from train_backbones import CNNClassifier

cam_layer = {
    "densenet161": "features[0].norm5",
    "densenet121": "features[0].norm5",
    "resnet34": "features[-2][1].bn2",
    "resnet18": "features[-2][1].bn2",
    "resnet50": "features[-2][1].bn2",
    "vgg19": "features[0][-3]"
}

cam_shape = {
    "densenet161": (7, 7),
    "densenet121": (7, 7),
    "resnet34": (7, 7),
    "resnet18": (7, 7),
    "resnet50": (7, 7),
    "vgg19": (14, 14)
}


def get_module(model, path):
    """Safely access a nested module using a dot-separated path with optional indexing."""
    module = model
    parts = re.split(r'(\[.*?\])', path)  # Split on brackets, keeping them
    for part in parts:
        if not part:
            continue
        if part.startswith('[') and part.endswith(']'):
            index = int(part[1:-1])
            module = module[index]
        else:
            module = getattr(module, part.lstrip("."))
    return module


@torch.inference_mode()
def eval_accuracy(model: nn.Module, dataloader: DataLoader, logger: Logger, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    concept_acc = BinaryAccuracy(multidim_average='global').to(device=device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, img_ids = batch
        attrs = torch.tensor([id_to_attributes[im_id] for im_id in img_ids.tolist()], dtype=torch.long, device=device)

        output, cpt_output = model.predict(images)

        concept_acc(F.sigmoid(cpt_output), attrs)

        predicted = torch.argmax(output, dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Concept Accuracy: {concept_acc.compute():.4f}")

    return acc


class MyGradCAM(GradCAM):
    target_size = (7, 7,)

    def __init__(self, model, target_layers, target_size: tuple[int, int] = (7, 7), reshape_transform=None):
        super(MyGradCAM, self).__init__(model, target_layers, reshape_transform)
        self.target_size = target_size

    def compute_cam_per_layer(self, input_tensor: torch.Tensor, targets: list[torch.nn.Module], eigen_smooth: bool) -> list[np.ndarray]:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, self.target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        return np.squeeze(cam_per_target_layer[0])


def compute_test_set_cams(model: nn.Module, model_name: str, save_dir: str | Path, data_dir: str | Path, device: torch.device | str = 'cuda'):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    t = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = Cub2011Eval(root=data_dir, transform=t)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_cams = []  # type: list[torch.Tensor]
    all_img_ids = []  # type: list[int]
    all_attributes = []  # type: list[list[int]]

    target_layers = [get_module(model=model, path=cam_layer[args.backbone])]

    with MyGradCAM(model=model, target_layers=target_layers, target_size=cam_shape[model_name]) as grad_cam:
        for im, _, img_id in tqdm(dataloader, total=len(dataloader)):
            im = im.to(device)
            sample_cams = []
            attrs = id_to_attributes[img_id.item()]  # type: list[int]

            for i, attr_i in enumerate(attrs):
                if attr_i == 0:
                    sample_cams.append(np.full(grad_cam.target_size, np.nan, dtype=np.float32))
                else:
                    targets = [ClassifierOutputTarget(i)]
                    attribute_cam = grad_cam(input_tensor=im, targets=targets)  # shape: (1, 224, 224)
                    sample_cams.append(attribute_cam)

            all_cams.append(torch.from_numpy(np.stack(sample_cams, axis=0)))  # shape: (n_attributes, 224, 224)
            all_img_ids.append(img_id.item())
            all_attributes.append(attrs)

    all_cams_pt = torch.stack(all_cams, dim=0)
    all_img_ids_pt = torch.tensor(all_img_ids)
    all_attributes_pt = torch.tensor(all_attributes)

    torch.save(
        dict(
            all_cams_pt=all_cams_pt,
            all_img_ids_pt=all_img_ids_pt,
            all_attributes_pt=all_attributes_pt
        ),
        Path(save_dir) / "cams.pth",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-cams", action="store_true")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--dataset", type=str, default="cub", choices=["cub", "celeba"])

    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "eval.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes, num_concepts = (200, 112 if args.dataset == "cub" else 8)

    # Load models
    model = CNNClassifier(args.backbone)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict=state_dict)
    
    proj_layer = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_concepts, bias=False)
    final_layer = torch.nn.Linear(num_concepts, 200)

    weight_b_g = torch.load(log_dir / "b_g.pt", map_location="cpu")
    weight_W_c = torch.load(log_dir / "W_c.pt", map_location="cpu")
    weight_W_g = torch.load(log_dir / "W_g.pt", map_location="cpu")

    proj_layer.weight.data = weight_W_c
    final_layer.weight.data = weight_W_g
    final_layer.weight.bias = weight_b_g

    model.classifier = nn.Sequential(proj_layer, final_layer)
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    t = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = Cub2011Eval(root=args.data_dir, transform=t)
    dataloader_test = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

    eval_accuracy(model=model, dataloader=dataloader_test, logger=logger, device=device)

    if args.compute_cams:
        compute_test_set_cams(
            model=model,
            model_name=args.backbone,
            save_dir=args.log_dir,
            data_dir=args.data_dir,
            device=device
        )

    cams = torch.load(Path(args.log_dir) / "cams.pth")
    all_cams_pt = cams["all_cams_pt"].to(device)
    all_attributes_pt = cams['all_attributes_pt'].to(device)

    n_attr = all_attributes_pt.size(1)
    all_activation_maps = []  # all_activation_maps[i] : (n_select_samples, fea_h, fea_w) for the i-th attribute

    for i in tqdm(range(n_attr)):
        attr_labels = all_attributes_pt[:, i]
        selected_img_indices = torch.nonzero(attr_labels == 1).squeeze()  # Select all the test images containing this attribute
        selected_cams = all_cams_pt[selected_img_indices][:, i]  # (n_select_samples, fea_h, fea_w)

        all_activation_maps.append(selected_cams)

    mean_loc_acc, _ = evaluate_concept_trustworthiness(all_activation_maps=all_activation_maps, all_img_ids=cams['all_img_ids_pt'])
    logger.info(f"Concept trustworthiness score: {mean_loc_acc:.2f}%")
