import torch
import torch.utils.data
import math
from torchvision import transforms

from .dataset import Dataset
from .alt_train import _loss


class AltEvaluator(object):
    def __init__(self, path_to_lmdb_dir, number_images_to_evaluate):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.dataset = Dataset(path_to_lmdb_dir, transform)
        if number_images_to_evaluate:
            self.dataset = self.dataset[0:int(number_images_to_evaluate)]
        self.batch_size = 1
        self._loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def evaluate(self, model):
        results = []

        with torch.no_grad():

            for batch_idx, (images, length_labels, digits_labels, paths) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cpu(), length_labels.cpu(), [digit_labels.cpu() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                print("Evaluating images in batch: ", batch_idx + 1)

                # Calculate loss for batch
                loss = _loss(length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits,
                             length_labels, digits_labels)

                # This only makes sense for batch size of 1
                batch_results = {}
                for image in paths:
                    batch_results[image.decode("utf-8")] = {"loss": loss.item()}

                results.append(batch_results)

        return results
