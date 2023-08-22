from pytorch/pytorch

workdir /workspace/ldm

copy . /workspace/ldm

run apt-get update && apt-get upgrade -y

run apt-get install git -y

run conda env create -f environment.yaml
