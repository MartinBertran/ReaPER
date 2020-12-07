FROM nvidia/cudagl:10.1-devel-ubuntu16.04

RUN apt-get update -yq \
    && apt-get upgrade -yq \
    && apt-get update -yq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common  libglfw3 \
    zlib1g-dev libjpeg-dev  libboost-all-dev \
    xvfb libav-tools xorg-dev libsdl2-dev \
    build-essential libxmu-dev libxi-dev libgl-dev libegl1-mesa-dev \
    sudo curl cmake software-properties-common  net-tools wget unzip  vim  virtualenv wget htop git ffmpeg swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.7
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.7 \
 && conda clean -ya



RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    pytorch=1.5.0 \
    torchvision=0.6.0 \
    pip \
    lockfile \
    && conda clean -ya

# Set the default command to python3
CMD ["python3"]



# Mujoco_py #Edit with your own instalation!!!!!!!
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200_linux/bin
ENV MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200_linux/
ENV MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
ENV MJLIB_PATH=$HOME/.mujoco/mujoco200_linux/bin/libmujoco200.so
ENV MJKEY_PATH=$HOME/.mujoco/mjkey.txt
ENV MUJOCO_GL=egl
#ENV MUJOCO_GL=osmesa

RUN echo 'export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200_linux/' >> /home/user/.bash_profile
RUN echo 'export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt' >> /home/user/.bash_profile
RUN echo 'export MJLIB_PATH=$HOME/.mujoco/mujoco200_linux/bin/libmujoco200.so' >> /home/user/.bash_profile
RUN echo 'export MJKEY_PATH=$HOME/.mujoco/mjkey.txt' >> /home/user/.bash_profile
RUN echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200_linux/bin:$LD_LIBRARY_PATH' >> /home/user/.bash_profile
RUN echo 'export PATH=/home/user/miniconda/bin:$PATH' >> /home/user/.bash_profile
RUN echo 'export MUJOCO_GL=egl' >> /home/user/.bash_profile
#RUN echo 'export MUJOCO_GL=osmesa' >> /home/user/.bash_profile


# For DM_Control...
RUN mkdir -p /home/user/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip -n mujoco.zip -d ~/.mujoco \
    && rm mujoco.zip


ARG MUJOCO_KEY
ENV MUJOCO_KEY=$MUJOCO_KEY
RUN echo $MUJOCO_KEY | base64 --decode > /home/user/.mujoco/mjkey.txt
ADD .mujoco/mjkey.txt $HOME/.mujoco/mjkey.txt


RUN pip install --upgrade pip Cython glfw
RUN pip install PyOpenGL PyOpenGL_accelerate
RUN git clone https://github.com/openai/mujoco-py.git; cd mujoco-py; python setup.py install ; cd ..
RUN git clone https://github.com/deepmind/dm_control.git; cd dm_control; pip install -r requirements.txt; python setup.py install ; cd ..
RUN git clone https://github.com/openai/gym; cd gym; pip install -e .[all]; cd ..
RUN pip install psutil  pyprind atari_py  pytest procgen opencv-python tqdm moviepy tensorboard plotly tensorboardX
RUN pip install scikit-image














