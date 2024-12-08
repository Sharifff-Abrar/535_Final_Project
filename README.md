# 535_Final_Project

## Ultra-Fast-Lane-Detection

### Installation

1. Clone the project:
    ```sh
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
    cd Ultra-Fast-Lane-Detection
    ```
2. Create a conda virtual environment and activate it:
    ```sh
    conda create -n lane-det python=3.7 -y
    conda activate lane-det
    ```
3. Install dependencies:
    ```sh
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 
    pip install -r requirements.txt
    ```

### Data Preparation

1. Download CULane and extract them to `$CULANEROOT`. The directory arrangement of CULane should look like:
    ```
    $CULANEROOT
    ├── driver_100_30frame
    ├── driver_161_90frame
    ├── driver_182_30frame
    ├── driver_193_90frame
    ├── driver_23_30frame
    ├── driver_37_30frame
    ├── laneseg_label_w16
    ├── list
    ```

### Training

1. Modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment.
2. For single GPU training, run:
    ```sh
    python train.py configs/path_to_your_config
    ```
3. For multi-GPU training, run:
    ```sh
    sh launch_training.sh
    or
    python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
    ```

### Visualization

1. To visualize the log with tensorboard, run:
    ```sh
    tensorboard --logdir log_path --bind_all
    ```
2. To visualize the detection results, run:
    ```sh
    python demo.py configs/culane.py --test_model path_to_culane_18.pth
    # or
    python demo.py configs/tusimple.py --test_model path_to_tusimple_18.pth
    ```

### Speed

1. To test the runtime, run:
    ```sh
    python speed_simple.py  
    # or
    python speed_real.py
    ```

## FENet: Focusing Enhanced Network for Lane Detection

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/HanyangZhong/FENet.git
    ```
2. Create a conda virtual environment and activate it:
    ```sh
    conda create -n fenet python=3.8 -y
    conda activate fenet
    ```
3. Install dependencies:
    ```sh
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
    # Or via pip
    pip install torch==1.11.0 torchvision==0.12.0
    pip install -r requirement.txt
    python setup.py build develop
    ```

### Training

1. For training, run:
    ```sh
    python main.py [configs/path_to_your_config] --gpus [gpu_num]
    # Example
    python main.py configs/fenet/FENetV1_dla34_culane.py --gpus 0
    ```

### Validation

1. For testing, run:
    ```sh
    python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
    # Example
    python main.py configs/fenet/FENetV2_dla34_culane.py --validate --load_from ./checkpoint/fenetv2_culane_dla34.pth --gpus 0
    ```

### Visualization

1. To get the visualization result, add `--view` when testing.

## License

This project is licensed under the MIT License.
