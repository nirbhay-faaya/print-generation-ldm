from pytorch/pytorch

workdir /workspace/ldm

copy . /workspace/ldm

run apt-get update && apt-get upgrade -y

run apt-get install git zip unzip vim -y

run conda env create -f environment.yaml

run conda init bash

run echo "conda activate ldm" >> ~/.bashrc

