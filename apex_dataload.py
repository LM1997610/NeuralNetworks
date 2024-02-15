
import torch
from torchvision import datasets
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision

import matplotlib.pyplot as plt 
import matplotlib.patches as patches

class ApexDetection(datasets.VisionDataset):

    def __init__(self, root: str, split = "train", transforms= None) -> None:

        super().__init__(root, transforms)
        self.split = split
        self.root = root
        self.ids = [i for i in range(len(self.root))]
        self.transforms = transforms

    def _load_image(self, idx: int):
      image = self.root[idx]['image']
      return image

    def _load_target(self, idx: int):
      
      target = self.root[idx]['objects.category']
      return target
    
    def _load_box(self, idx: int):
      bbox = self.root[idx]['objects.bbox']
      return bbox

    def __getitem__(self, idx:int):
      image = self._load_image(idx)
      bbox = self._load_box(idx)
      label = self._load_target(idx)

      bbox = tv_tensors.BoundingBoxes(bbox, format="XYWH", canvas_size=F.get_size(image))
      bbox = torchvision.ops.box_convert(bbox, in_fmt="xywh", out_fmt="xyxy")
      
      label = torch.tensor(label, dtype=torch.int64)
      image = tv_tensors.Image(image)
      image = image.float() / 255.0

      iscrowd = torch.zeros((len(label),), dtype=torch.int64)

      target = {}
      target["boxes"] = bbox
      target["labels"] = label
      target["image_id"] = torch.tensor(self.root[idx]['image_id'], dtype=torch.int64)
      target["area"] = torch.tensor(self.root[idx]['objects.area'], dtype=torch.int64)
      target["iscrowd"]= iscrowd

      if self.transforms is not None:
            img, target = self.transforms(img, target)

      return image, target

    def __len__(self):
        return len(self.ids)

    def do_plot(self, idx:int, ax=None):
      
      image, target =self. __getitem__(idx)
      boxes, labels, image_id = target['boxes'], target['labels'], target['image_id']

      if ax is None:
        ax = plt.gca()

      ax.set_axis_off()

      for box, label,  in zip(boxes, labels):
        x,y,w,h = box
        color = '#FFDD33' if label == 1 else 'blue'
        rect = patches.Rectangle((x, y), w-x, h-y, linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
      ax.set_title("Image_id: "+str(image_id.detach().numpy()), fontsize=8, loc='left')
      bp = ax.imshow(image.squeeze(0).permute(1, 2, 0), aspect="auto")

      return bp
      