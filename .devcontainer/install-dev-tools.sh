# update system
apt-get update
apt-get upgrade -y
# install Linux tools and Python 3
apt update
apt install -y --no-install-recommends build-essential ca-certificates ccache cmake curl git libjpeg-dev nano libpng-dev python3.10 python3.10-dev python3-pip 
apt install -y python3.10-distutils 
ln -s /usr/bin/python3.10 /usr/local/bin/python
rm -rf /var/lib/apt/lists/*
# install Python packages
python3 -m pip install --upgrade pip
python3.10 -m pip install --upgrade pip setuptools 
python3.10 -m pip install astunparse expecttest hypothesis numpy psutil pyyaml requests setuptools types-dataclasses typing-extensions sympy filelock networkx jinja2 fsspec protobuf 
python3.10 -m pip install -r requirements.txt 
python3.10 -m pip install torch torchvision torchaudio
# update CUDA Linux GPG repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb
# install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt-get update
apt-get install libcudnn8=8.9.0.*-1+cuda11.8
apt-get install libcudnn8-dev=8.9.0.*-1+cuda11.8
# install recommended packages
apt-get install zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y
# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean