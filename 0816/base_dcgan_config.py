import torch
# base_dcgan_config

nz = 100 # 잠재공간 벡터의 크기

ngf = 64 # 생성자를 통과하는 특징 데이터들의 채널 크기

ndf = 64 # 판별자를 통과하는 특징 데이터들의 채널 크기

nc = 3 # RGB 이미지이기 때문에 3으로 설정합니다.


workers = 2
batch_size = 64
image_size = 64
num_epochs = 50
lr = 0.00005

beta1 = 0.4

device = 'cuda' if torch.cuda.is_available() else 'cpu'