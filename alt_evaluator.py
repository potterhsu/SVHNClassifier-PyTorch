import torch
import torch.utils.data
from torchvision import transforms

from dataset import Dataset


class AltEvaluator(object):
    def __init__(self, path_to_lmdb_dir):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir, transform), batch_size=128, shuffle=False)

    def evaluate(self, model):
        with torch.no_grad():
            for _, (images, length_labels, digits_labels) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cpu(), length_labels.cpu(), [digit_labels.cpu() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                # length
                length_predictions = length_logits.max(1)[1].tolist()
                length_logits_list = length_logits.tolist()
                length_results = zip(length_predictions,length_logits_list)
                
                # digit1
                digit1_predictions = digit1_logits.max(1)[1].tolist()
                digit1_logits_list = digit1_logits.tolist()
                digit1_results = zip(digit1_predictions,digit1_logits_list)
                
                # digit2
                digit2_predictions = digit2_logits.max(1)[1].tolist()
                digit2_logits_list = digit2_logits.tolist()
                digit2_results = zip(digit2_predictions,digit2_logits_list)

                # digit3
                digit3_predictions = digit3_logits.max(1)[1].tolist()
                digit3_logits_list = digit3_logits.tolist()
                digit3_results = zip(digit3_predictions,digit3_logits_list)

                # digit4
                digit4_predictions = digit4_logits.max(1)[1].tolist()
                digit4_logits_list = digit4_logits.tolist()
                digit4_results = zip(digit4_predictions,digit4_logits_list)

                # digit5
                digit5_predictions = digit5_logits.max(1)[1].tolist()
                digit5_logits_list = digit5_logits.tolist()
                digit5_results = zip(digit5_predictions,digit5_logits_list)
                
        results = {
            "length": length_results,
            "digit1": digit1_results, 
            "digit2": digit2_results,
            "digit3": digit3_results,
            "digit4": digit4_results,
            "digit5": digit5_results,
        }
        return results
