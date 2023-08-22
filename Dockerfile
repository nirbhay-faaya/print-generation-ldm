from pytorch/pytorch

workdir /workspace

copy . /workspace

run apt-get update && apt-get upgrade -y

run apt-get install git -y

run pip install -r requirements.txt
