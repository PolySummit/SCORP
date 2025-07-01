# /usr/bin/bash
# This script sets up the environment for the gsobj project.

git submodule update --init --recursive

conda create -n gsobj python=3.11 -y
conda activate gsobj
conda install matplotlib black jupyter tqdm pandas scikit-image numpy\<2 -y
conda install faiss-gpu==1.8.0 \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    pytorch-cuda=11.8 \
    cuda-toolkit \
    -c pytorch \
    -c nvidia/label/cuda-11.8.0 \
    -y


cd submodules/sam2/
pip install -e .
pip install -e ".[notebooks]"


cd ../GroundingDINO/
pip install -e .


cd ../mast3r/
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt

pip install cython
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .
cd ..

cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../


cd ../simple-knn/
pip install -e .

cd ../diff-gaussian-rasterization/
pip install -e .

cd ../diff-surfel-rasterization/
pip install -e .


cd ..

pip install plyfile open3d e3nn