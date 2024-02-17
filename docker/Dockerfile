FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

## Base packages for ubuntu
# clean the libs list
RUN apt-get clean \
 && apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    curl \
    bzip2 \
    vim \
    nano \
    g++ \
    make \
    cmake \
    build-essential \
    software-properties-common \
    sudo \
    gosu \
    libx11-6 \
    graphviz \
    tmux \
    screen \
    htop \
    p7zip-full \
    locales \
&& rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

## Download and install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && echo "export PATH=/home/user/miniconda/bin:$PATH" >>/home/user/.profile
ENV PATH /home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.8 environment.
COPY --chown=user docker/environment.yml /opt/environment.yml
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda env create -f /opt/environment.yml \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=mesh-diffusion
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

ENV PATH=/usr/local/cuda/bin:$PATH
ENV CPATH=/usr/local/cuda/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

ENV CUDA=cu118
ENV TORCH=2.1.1

COPY --chown=user docker/requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt
RUN pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

ENV WANDB_CACHE_DIR=/code/.cache/wandb
ENV WANDB_CACHE_DIR=/code/.config/wandb
