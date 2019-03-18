#include <torch/script.h>
#include <torch/utils.h>

#include <opencv2/opencv.hpp>


void _infer(const std::string& pathToCheckpointFile, const std::string& pathToInputImage) {
    std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(pathToCheckpointFile);
    model->to(torch::kCUDA);

    torch::NoGradGuard noGradGuard;

    cv::Mat image;
    image = cv::imread(pathToInputImage, CV_LOAD_IMAGE_COLOR);

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(64, 64));
    image = image(cv::Rect((64 - 54) / 2, (64 - 54) / 2, 54, 54)).clone();  // it's necessary to clone the ROI image to make elements continuous

    torch::Tensor images_tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kByte).unsqueeze(0);
    images_tensor = images_tensor.permute({0, 3, 1, 2});  // BHWC to BCHW
    images_tensor = images_tensor.toType(torch::kFloat);
    images_tensor = images_tensor.div(255);
    images_tensor = images_tensor.sub(0.5).div(0.5);
    images_tensor = images_tensor.to(torch::kCUDA);

    auto outputs = model->forward({images_tensor}).toTuple()->elements();
    torch::Tensor length_logits = outputs[0].toTensor();
    torch::Tensor digit1_logits = outputs[1].toTensor();
    torch::Tensor digit2_logits = outputs[2].toTensor();
    torch::Tensor digit3_logits = outputs[3].toTensor();
    torch::Tensor digit4_logits = outputs[4].toTensor();
    torch::Tensor digit5_logits = outputs[5].toTensor();

    auto length_prediction = std::get<1>(length_logits.max(1)).item<int>();
    auto digit1_prediction = std::get<1>(digit1_logits.max(1)).item<int>();
    auto digit2_prediction = std::get<1>(digit2_logits.max(1)).item<int>();
    auto digit3_prediction = std::get<1>(digit3_logits.max(1)).item<int>();
    auto digit4_prediction = std::get<1>(digit4_logits.max(1)).item<int>();
    auto digit5_prediction = std::get<1>(digit5_logits.max(1)).item<int>();

    printf("length: %d \n", length_prediction);
    printf("digits: %d %d %d %d %d \n", digit1_prediction, digit2_prediction, digit3_prediction, digit4_prediction, digit5_prediction);
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: infer <path-to-checkpoint-file> <path-to-input-image> \n";
        return -1;
    }

    const std::string pathToCheckpointFile = argv[1];
    const std::string pathToInputImage = argv[2];

    _infer(pathToCheckpointFile, pathToInputImage);

    return 0;
}