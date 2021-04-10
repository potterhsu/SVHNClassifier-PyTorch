import argparse
import base64
import concurrent
import io

from PIL import Image
import grpc
import numpy as np
import torch
import torchvision

import infer as prediction_module
import model
import svhn_classifier_pb2
import svhn_classifier_pb2_grpc


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SvhnClassifierServicer(svhn_classifier_pb2_grpc.SvhnClassifierServicer):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = model.Model()
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=device))
        self.model.to(device)

    def __predict(self, image_bytes, center_crop=True):
        with torch.no_grad():
            pipeline = [torchvision.transforms.Resize([64, 64])]
            if center_crop:
                pipeline.append(torchvision.transforms.CenterCrop([54, 54]))
            pipeline.append(torchvision.transforms.ToTensor())
            pipeline.append(torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
            transform = torchvision.transforms.Compose(pipeline)
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            image = transform(image)
            images = image.unsqueeze(dim=0).to(device)

            length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = self.model.eval()(images)

            length_prediction = length_logits.max(1)[1]
            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]
            digit3_prediction = digit3_logits.max(1)[1]
            digit4_prediction = digit4_logits.max(1)[1]
            digit5_prediction = digit5_logits.max(1)[1]

            length = length_prediction.item()
            digits = [
                digit1_prediction.item(),
                digit2_prediction.item(),
                digit3_prediction.item(),
                digit4_prediction.item(),
                digit5_prediction.item(),
            ]

            print(length, digits)
            return length, digits

    def Predict(self, request, context):
        length, digits = self.__predict(request.data, request.center_crop)
        return svhn_classifier_pb2.Prediction(length=length, digits=digits)


def serve(args):
    server = grpc.server(thread_pool=concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers))
    svhn_classifier_pb2_grpc.add_SvhnClassifierServicer_to_server(servicer=SvhnClassifierServicer(args.checkpoint), server=server)
    server.add_insecure_port("%s:%s" % (args.host, args.port))
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--max_workers", type=int, default=1, help="the number of max workers")
    args = parser.parse_args()
    serve(args)
