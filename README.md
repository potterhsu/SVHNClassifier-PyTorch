# SVHNClassifier-PyTorch

A PyTorch implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf) 


## Results

### Accuracy
![Accuracy](https://github.com/potterhsu/SVHNClassifier-PyTorch/blob/master/images/accuracy.png?raw=true)

> Accuracy 95.32% on test dataset after 721,000 steps

## Requirements

* Python 2.7
* PyTorch
* h5py

    ```
    In Ubuntu:
    $ sudo apt-get install libhdf5-dev
    $ sudo pip install h5py
    ```

* Protocol Buffers 3
* LMDB
* Visdom

## Setup

1. Clone the source code

    ```
    $ git clone https://github.com/potterhsu/SVHNClassifier-PyTorch
    $ cd SVHNClassifier-PyTorch
    ```

2. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

3. Extract to data folder, now your folder structure should be like below:
    ```
    SVHNClassifier
        - data
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
    ```


## Usage

1. (Optional) Take a glance at original images with bounding boxes

    ```
    Open `draw_bbox.ipynb` in Jupyter
    ```

1. Convert to LMDB format

    ```
    $ python convert_to_lmdb.py --data_dir ../data
    ```

1. (Optional) Test for reading LMDBs

    ```
    Open `read_lmdb_sample.ipynb` in Jupyter
    ```

1. Train

    ```
    $ python train.py --data_dir ../data --logdir ./logs
    ```

1. Retrain if you need

    ```
    $ python train.py --data_dir ./data --logdir ./logs_retrain --restore_checkpoint ./logs/model-100.tar
    ```

1. Evaluate

    ```
    $ python eval.py --data_dir ./data ./logs/model-100.tar
    ```

1. Visualize

    ```
    $ python -m visdom.server
    $ python visualize.py --logdir ./logs
    ```

1. Clean

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs_retrain
    ```
