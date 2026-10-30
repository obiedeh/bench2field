# Isaac ROS 4.5 (ROS 2 Jazzy) from NVIDIA's published apt packages, in a plain
# Ubuntu 24.04 container. This is the install the Isaac ROS getting-started
# guide performs on a host, done inside a container so the board itself is
# not changed. Nothing is built from source; every package comes signed from
# the four repositories below.
#
#   x86_64 workstation:  docker build -t isaac-ros-4.5:x86_64 .
#   Jetson AGX Thor:     docker build --build-arg ARCH=sbsa --build-arg SUITE=noble-jetpack \
#                                     --build-arg JETSON_REPO=common -t isaac-ros-4.5:thor .
#
# Run with the NVIDIA container runtime (--gpus all or --runtime nvidia).
FROM ubuntu:24.04

ARG ARCH=x86_64
ARG SUITE=noble
ARG JETSON_REPO=x86_64/noble
ARG ISAAC_ROS_RELEASE=release-4.5
ARG L4T_RELEASE=r38.4

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release wget \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 apt source (packages.ros.org key and list, as the ROS 2 Jazzy install guide does)
RUN curl -fsSL -o /tmp/ros2-apt-source.deb \
        https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb \
    && apt-get update && apt-get install -y /tmp/ros2-apt-source.deb && rm /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# Isaac ROS apt repository
RUN curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | gpg --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/${ISAAC_ROS_RELEASE} ${SUITE} main" \
        > /etc/apt/sources.list.d/nvidia-isaac-ros.list

# CUDA apt repository (CUDA runtime libraries and TensorRT)
RUN wget -q -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && rm -rf /var/lib/apt/lists/*

# Jetson apt repository (VPI, NvSci and the other L4T user-space libraries Isaac ROS links against)
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/${JETSON_REPO} ${L4T_RELEASE} main" \
        > /etc/apt/sources.list.d/nvidia-jetson.list

# ROS 2 Jazzy base, the Isaac ROS object detection quickstart packages, and rosbag2 for the quickstart bag
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-isaac-ros-yolov8 \
        ros-jazzy-isaac-ros-dnn-image-encoder \
        ros-jazzy-isaac-ros-tensor-rt \
        ros-jazzy-isaac-ros-examples \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-image-transport \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:${PATH}
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc
SHELL ["/bin/bash", "-c"]
CMD ["bash"]
