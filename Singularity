Bootstrap: docker
From: ubuntu

%help
Singularity container for auto-sklearn

%labels
    Version v0.1

%environment
    export PATH=/data/miniconda/bin:$PATH

%setup
    mkdir ${SINGULARITY_ROOTFS}/data

%post
    apt-get update
    apt-get -y install wget git gcc g++ tar libpython3.6-dev
    cd /data
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/miniconda
    export PATH=/data/miniconda/bin:$PATH

    conda install --yes -c anaconda gcc_linux-64 gxx_linux-64
    conda install --yes pip wheel setuptools

    git clone https://github.com/urbanmatthias/tpot.git
    cd tpot

    conda install numpy scipy scikit-learn pandas
    pip install deap update_checker tqdm stopit openml xgboost
    python setup.py install
