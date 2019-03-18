import torch
import torch.utils.data
from torchvision import transforms

from dataset import Dataset


class Evaluator(object):
    def __init__(self, path_to_lmdb_dir):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir, transform), batch_size=128, shuffle=False)

    def evaluate(self, model):
        num_correct = 0
        needs_include_length = False

        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                length_prediction = length_logits.max(1)[1]
                digit1_prediction = digit1_logits.max(1)[1]
                digit2_prediction = digit2_logits.max(1)[1]
                digit3_prediction = digit3_logits.max(1)[1]
                digit4_prediction = digit4_logits.max(1)[1]
                digit5_prediction = digit5_logits.max(1)[1]

                if needs_include_length:
                    num_correct += (length_prediction.eq(length_labels) &
                                    digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3]) &
                                    digit5_prediction.eq(digits_labels[4])).cpu().sum()
                else:
                    num_correct += (digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3]) &
                                    digit5_prediction.eq(digits_labels[4])).cpu().sum()

        accuracy = num_correct.item() / len(self._loader.dataset)
        return accuracy
