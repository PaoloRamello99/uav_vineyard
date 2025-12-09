FROM althack/ros2:jazzy-full

WORKDIR /workspaces

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    wget \
    python3-tk \
    python3-setuptools \
    python3-venv \
    ros-jazzy-plotjuggler-ros \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ffmpeg \
    tree \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Create virtualenv for JAX and Python packages
# -------------------------------------------------------------
RUN python3 -m venv /opt/jax-env && \
    /opt/jax-env/bin/pip install --upgrade pip && \
    /opt/jax-env/bin/pip install jax

# Make venv Python first
ENV PATH="/opt/jax-env/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/jax-env"

# -------------------------------------------------------------
# PX4
# -------------------------------------------------------------
RUN git clone https://github.com/PX4/PX4-Autopilot.git --branch=main --recursive
RUN bash ./PX4-Autopilot/Tools/setup/ubuntu.sh

# -------------------------------------------------------------
# XRCE-DDS Agent
# -------------------------------------------------------------
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && ldconfig && \
    cd /workspaces && rm -rf Micro-XRCE-DDS-Agent

RUN chown -R ros:ros /workspaces

USER ros

# Activate venv + ROS + workspace
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/ros/.bashrc && \
    echo "source /opt/jax-env/bin/activate" >> /home/ros/.bashrc && \
    echo "if [ -f /workspaces/uav_vineyard/install/setup.bash ]; then source /workspaces/uav_vineyard/install/setup.bash; fi" >> /home/ros/.bashrc

CMD ["/bin/bash"]
