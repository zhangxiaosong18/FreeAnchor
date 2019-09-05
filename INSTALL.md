## Installation

### Requirements:
- Python3
- PyTorch 1.1 with CUDA support
- torchvision 0.2.1
- pycocotools
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name free_anchor python=3.7
conda activate free_anchor

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# pytorch and torchvision
# we give the instructions for CUDA 10.0
conda install pytorch=1.1 torchvision=0.2.1 cudatoolkit=10.0 -c pytorch

# install pycocotools
pip install pycocotools

# install FreeAnchor
git clone https://github.com/zhangxiaosong18/FreeAnchor.git

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
cd FreeAnchor
python setup.py build develop

# or if you are on macOS
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```
