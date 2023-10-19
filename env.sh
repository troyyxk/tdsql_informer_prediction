# wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2019.03-Linux-x86_64.sh
# conda create -n xxie python=3.6.10
# conda activate xxie


conda install -y  pytorch==1.4.0 torchvision==0.5.0 -c pytorch &&
pip install numpy==1.18.5 tensorboardX==2.0 tqdm==4.45.0 scikit-image==0.16.2 scikit-learn==0.23.2 albumentations==0.5.0 tensorflow-gpu==2.3.1 prefetch_generator==1.0.1 seaborn==0.11.1
