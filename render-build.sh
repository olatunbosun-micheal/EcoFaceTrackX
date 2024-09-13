# render-build.sh
apt-get update && apt-get install -y \
libsm6 libxext6 libxrender-dev \
libopencv-dev python3-opencv \
build-essential cmake \
&& pip install -r requirements.txt
