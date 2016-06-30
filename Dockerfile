# Start with cuDNN base image
FROM nvidia/cuda:7.5-cudnn5-devel
MAINTAINER SeanBE

# Install git, wget, python-dev, pip and other dependencies
RUN apt-get update && apt-get install -y \
  git \
  wget \
  libopenblas-dev \
  python-dev \
  python-pip \
  python-nose \
  python-numpy \
  python-scipy \
  libhdf5-dev \
  python-h5py \
  python-yaml

RUN pip install --upgrade six

RUN apt-get update --fix-missing

RUN apt-get install --yes build-essential cmake git pkg-config libjpeg8-dev \
                          libtiff5 libjasper-dev libpng12-dev \
                          libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
                          libv4l-dev python2.7-dev

RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y unzip  \
	 	libswscale-dev \
		python3-dev python3-numpy \
		libtbb2 libtbb-dev libjpeg-dev \
		libpng-dev libtiff-dev libjasper-dev

RUN cd \
	&& wget https://github.com/Itseez/opencv/archive/3.1.0.zip \
	&& unzip 3.1.0.zip \
	&& cd opencv-3.1.0 \
	&& mkdir build \
	&& cd build \
	&& cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_CUDA=OFF .. \
	&& make -j3 \
	&& make install \
	&& cd \
	&& rm 3.1.0.zip

ENV CUDA_ROOT /usr/local/cuda/bin
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=1\n[nvcc]\nfastmath=True" > /root/.theanorc

RUN cd /root && git clone https://github.com/fchollet/keras.git && cd keras && python setup.py install
