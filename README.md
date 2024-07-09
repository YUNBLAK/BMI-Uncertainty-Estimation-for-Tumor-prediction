# Uncertainty Estimation for Tumor Prediction

# Setup
Install the [docker](https://www.docker.com/). Here are some guides for install docker for [Ubuntu](https://docs.docker.com/desktop/install/mac-install/) and docker for [MacOS](https://www.docker.com/)

# Dependencies


More details are in file environment.yaml

## NVIDIA Container Toolkit
Installation of the NVIDIA Container Toolkit: To use the GPU in Docker, the NVIDIA Container Toolkit must be installed. Install using the following command:

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker




Clone this repository to your local workspace:

    $ git https://github.com/YUNBLAK/BMI-Uncertainty-Estimation-for-Tumor-prediction.git

This contains a Dockerfile that can be used to build and test our implementation. Go to the directory where the docker file exists and run the command below. If permission denied occurs, run docker in root sudo:

    $ sudo docker build -t my-torch-app .

Build without any cashes:
    
    $ sudo docker build --no-cache -t my-torch-app .



## Examples for Container Usage


# Citation

    @InProceedings{Yun_2024_CVPR,
        author    = {Yun, Juyoung and Abousamra, Shahira and Li, Chen and Gupta, Rajarsi and Kurc, Tahsin and Samaras, Dimitris and Van Dyke, Alison and Saltz, Joel and Chen, Chao},
        title     = {Uncertainty Estimation for Tumor Prediction with Unlabeled Data},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2024},
        pages     = {6946-6954}
    }
