FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Update and install sudo and dependencies
RUN apt-get update && apt-get -y install sudo

# Install Python 3.10 and openslide in non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-openslide libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to use python3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip==21.0.1 && \
    pip install -U pip setuptools && \
    pip install setuptools==45 && \
    pip install cython && \
    pip install \
        contourpy==1.2.1 \
        cycler==0.12.1 \
        filelock==3.14.0 \
        fonttools==4.53.0 \
        fsspec==2024.3.1 \
        imageio==2.34.1 \
        jinja2==3.1.3 \
        joblib==1.4.0 \
        kiwisolver==1.4.5 \
        lazy-loader==0.4 \
        markupsafe==2.1.5 \
        matplotlib==3.9.0 \
        mpmath==1.3.0 \
        networkx==3.3 \
        numpy==1.26.4 \
        opencv-python==4.9.0.80 \
        openslide-python==1.3.1 \
        packaging==24.0 \
        pandas==2.2.2 \
        pillow==10.3.0 \
        pyparsing==3.1.2 \
        python-dateutil==2.9.0.post0 \
        pytz==2024.1 \
        scikit-image==0.23.2 \
        scikit-learn==1.4.2 \
        scipy==1.13.0 \
        six==1.16.0 \
        sympy==1.12 \
        threadpoolctl==3.5.0 \
        tifffile==2024.5.10 \
        torch==2.3.0 \
        torchvision==0.18.0 \
        tqdm==4.66.2 \
        triton==2.3.0 \
        typing-extensions==4.11.0 \
        tzdata==2024.1

COPY . .

RUN chmod +x save_svs_to_tiles.sh start_con.sh start.sh auto.sh

# Use ENTRYPOINT to set the default executable
ENTRYPOINT ["bash"]