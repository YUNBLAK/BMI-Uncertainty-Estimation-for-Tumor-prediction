# Uncertainty Estimation for Tumor Prediction

# Setup
**1. Docker:**
Install the [docker](https://www.docker.com/). Here are some guides for install docker for [Ubuntu](https://docs.docker.com/desktop/install/mac-install/) and docker for [MacOS](https://www.docker.com/)

**2. Download Pretrained Model:**
Please download two pretrained model, [Uncertainty Model](https://drive.google.com/file/d/1YlSQzmeggKB97pATNdlmeq6Q7PUr3EW-/view?usp=sharing) and [Baseline Model](https://drive.google.com/file/d/10H7Ce79zfb1b3tdX5dNexjIqKkCLtWzN/view?usp=sharing), and move them into model directory. 

**3. Move Whole Slide Image:**
Please move WSIs to dataset folder.

# Dependencies

    scikit-learn==1.4.2
    openslide-python==1.3.1
    opencv-python==4.9.0.80
    torch==2.3.0
    torchvision==0.18.0


More details are in file environment.yaml

## NVIDIA Container Toolkit
Installation of the NVIDIA Container Toolkit: To use the GPU in Docker, the NVIDIA Container Toolkit must be installed. Install using the following command:

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker



# How to Use Docker
Clone this repository to your local workspace:

    $ git https://github.com/YUNBLAK/BMI-Uncertainty-Estimation-for-Tumor-prediction.git

This contains a Dockerfile that can be used to build and test our implementation. Go to the directory where the docker file exists and run the command below. If permission denied occurs, run docker in root sudo:

    $ sudo docker build -t my-torch-app .

Build without any cashes if you want to re-build:
    
    $ sudo docker build --no-cache -t my-torch-app .


## Run docker image

    $ sudo docker run --gpus all -it --rm -v "$PWD:/app" --entrypoint /bin/bash my-torch-app    
    $ sudo docker run --gpus all -it --rm my-torch-app


1.Extracting patches from Whole Slides Image:

    $ sudo docker run --gpus all -it --rm -v "$PWD:/app" my-torch-app save_svs_to_tiles.sh

2.Prediction with Uncertainty Model:

    $ sudo docker run --gpus all -it --rm -v "$PWD:/app" my-torch-app start_con.sh

## Examples for Container Usage
    
    $ sudo docker run --gpus all -it --rm -v "$PWD:/app" my-torch-app save_svs_to_tiles.sh





## Trouble Shooting:
**If you have any issue like:**
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.
Re-install docker:

    $ sudo apt-get purge -y docker-engine docker docker.io docker-ce docker-ce-cli
    $ sudo apt-get autoremove -y --purge docker-engine docker docker.io docker-ce
    $ sudo rm -rf /var/lib/docker /etc/docker
    $ sudo rm /etc/apparmor.d/docker
    $ sudo groupdel docker
    $ sudo rm -rf /var/run/docker.sock
    $ sudo apt-get install docker-ce docker-ce-cli containerd.io

## Extra-Commands:

Connect

# Citation

    @InProceedings{Yun_2024_CVPR,
        author    = {Yun, Juyoung and Abousamra, Shahira and Li, Chen and Gupta, Rajarsi and Kurc, Tahsin and Samaras, Dimitris and Van Dyke, Alison and Saltz, Joel and Chen, Chao},
        title     = {Uncertainty Estimation for Tumor Prediction with Unlabeled Data},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2024},
        pages     = {6946-6954}
    }
