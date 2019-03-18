# SVHNClassifier Inference in C++ using LibTorch

## Requirements

* libtorch 1.0
* OpenCV 3.4.3
    ```
    $ wget https://github.com/opencv/opencv/archive/3.4.3.zip -O opencv-3.4.3.zip
    $ unzip opencv-3.4.3.zip
    $ cd opencv-3.4.3
    $ mkdir build; cd build
    $ cmake ..
    $ make -j4
    $ sudo make install
    ```


## Setup

1. Download LibTorch and extract

1. Declare environment variable `Torch_DIR`
    ```
    $ export Torch_DIR=/path/to/libtorch
    ```
    > To tell `find_package` in `CMakeLists.txt` where to find LibTorch
    
1. Build project
    ```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    ```

1. Serializing script module to a file
    ```
    $ cd /path/to/project-root
    $ python
    >>> from model import Model
    >>> model = Model()
    >>> model.restore('./logs/model-54000.pth')
    >>> model.eval().save('model.pt')
    $ mv ./model.pt ./cpp/
    ```

## Usage

* Infer
    ```
    $ cd build
    $ ./infer ../model.pt ../../images/test-75.png
    ```
