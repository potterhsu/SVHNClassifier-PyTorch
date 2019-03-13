import argparse
import torch

from PIL import Image
from torchvision import transforms

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pth')
parser.add_argument('input', type=str, help='path to input image')


def _infer(path_to_checkpoint_file, path_to_input_image):
    model = Model()
    model.load(path_to_checkpoint_file)
    model.cuda()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(dim=0).cuda()

        length_logits, digits_logits = model(images)

        length_predictions = length_logits.max(1)[1]
        digits_predictions = [digit_logits.max(1)[1] for digit_logits in digits_logits]

        print('length:', length_predictions.item())
        print('digits:', [it.item() for it in digits_predictions])


def main(args):
    path_to_checkpoint_file = args.checkpoint
    path_to_input_image = args.input

    _infer(path_to_checkpoint_file, path_to_input_image)


if __name__ == '__main__':
    main(parser.parse_args())
