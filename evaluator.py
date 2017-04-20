import torch.utils.data
from torch.autograd import Variable
from dataset import Dataset


class Evaluator(object):
    def __init__(self, path_to_lmdb_dir):
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir), batch_size=128, shuffle=False)

    def evaluate(self, model):
        model.eval()
        num_correct = 0
        needs_include_length = False

        for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
            images, length_labels, digits_labels = (Variable(images.cuda(), volatile=True),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model(images)
            length_predictions = length_logits.data.max(1)[1]
            digits_predictions = [digit_logits.data.max(1)[1] for digit_logits in digits_logits]

            if needs_include_length:
                num_correct += (length_predictions.eq(length_labels.data) &
                                digits_predictions[0].eq(digits_labels[0].data) &
                                digits_predictions[1].eq(digits_labels[1].data) &
                                digits_predictions[2].eq(digits_labels[2].data) &
                                digits_predictions[3].eq(digits_labels[3].data) &
                                digits_predictions[4].eq(digits_labels[4].data)).cpu().sum()
            else:
                num_correct += (digits_predictions[0].eq(digits_labels[0].data) &
                                digits_predictions[1].eq(digits_labels[1].data) &
                                digits_predictions[2].eq(digits_labels[2].data) &
                                digits_predictions[3].eq(digits_labels[3].data) &
                                digits_predictions[4].eq(digits_labels[4].data)).cpu().sum()

        accuracy = num_correct / float(len(self._loader.dataset))
        return accuracy
