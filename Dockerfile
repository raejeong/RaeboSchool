FROM nvidia/cuda:8.0-cudnn6-devel

# nvidia-docker hooks
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# set up environment
ENV DEBIAN_FRONTEND noninteractive

# update repos/ppas...
RUN apt-get update 
RUN apt-get install -y python-software-properties software-properties-common curl
RUN apt-add-repository -y ppa:x2go/stable
RUN apt-get update 

# install core packages
RUN apt update
RUN apt-get install -y python3-pip git
RUN apt-get install -y python3-matplotlib python3-scipy python3-numpy

# install python packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade ipython[all]
RUN pip3 install pyyaml
RUN pip3 install opencv-python
RUN export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
RUN export CUDA_HOME=/usr/local/cuda
RUN pip3 install tensorflow-gpu
# set up gnuradio and related toolsm
RUN apt-get install -y sudo

# pytorch
RUN pip3 install -U numpy
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip3 install torchvision

# ROS
# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-core=1.3.1-0* \
    && rm -rf /var/lib/apt/lists/*

# Set up environment
RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

# User
RUN useradd -ms /bin/bash user
RUN echo 'user:1234' | chpasswd
RUN chown user /home/user
RUN usermod -a -G sudo user

# workspace
RUN mkdir /home/user/workspace
WORKDIR /home/user/

# Gym deps
RUN apt update 
RUN apt install -y python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig pypy-dev automake autoconf libtool

# set up OpenAI Gym
RUN git clone https://github.com/openai/gym.git && cd gym && pip3 install -e .
RUN pip3 install gym[all]


# Roboschool
RUN git clone https://github.com/openai/roboschool.git
RUN echo 'export ROBOSCHOOL_PATH=/home/user/roboschool:${ROBOSCHOOL_PATH}' >> ~/.bashrc
RUN apt update 
RUN apt install -y cmake ffmpeg pkg-config qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev libboost-python-dev libtinyxml-dev
RUN pip3 install pyopengl
WORKDIR /home/user/roboschool/
RUN git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision
RUN mkdir bullet3/build
WORKDIR bullet3/build
ENV ROBOSCHOOL_PATH /home/user/roboschool
RUN cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..
RUN make -j4
RUN make install
WORKDIR /home/user/roboschool/
ENV ROBOSCHOOL_PATH /home/user/roboschool
RUN pip3 install -e $ROBOSCHOOL_PATH
RUN chown -R user /home/user

RUN apt install -y gedit
RUN pip3 install --upgrade scipy

USER user
WORKDIR /home/user/
RUN echo 'export PYTHONPATH=/home/user/workspace:${PYTHONPATH}' >> ~/.bashrc
