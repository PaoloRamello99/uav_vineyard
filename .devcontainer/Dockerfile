FROM althack/ros2:jazzy-full

WORKDIR /workspaces

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    wget \
    python3-tk \
    python3-setuptools \
    ros-jazzy-plotjuggler-ros \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ffmpeg \
    tree \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip config set global.break-system-packages true

RUN git clone https://github.com/PX4/PX4-Autopilot.git --branch=main --recursive
RUN bash ./PX4-Autopilot/Tools/setup/ubuntu.sh

RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && mkdir build && cd build && \
    cmake .. && make && make install && ldconfig && \
    cd /workspaces && rm -rf Micro-XRCE-DDS-Agent

RUN chown -R ros:ros /workspaces

USER ros

RUN echo "\nsource /opt/ros/jazzy/setup.bash" >> /home/ros/.bashrc && \
    echo "if [ -f /workspaces/uav_vineyard/install/setup.bash ]; then source /workspaces/uav_vineyard/install/setup.bash; fi" >> /home/ros/.bashrc

CMD ["/bin/bash"]