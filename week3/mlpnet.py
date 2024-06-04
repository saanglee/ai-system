class MLPnet(nn.Module):    # nn.Module을 상속받음
    def __init__(self):     # 생성자 정의 (해당 클래스의 인스턴스가 생성될 때 자동으로 호출)
        super(MLPnet, self).__init__()    # nn.Module의 생성자__init__() 을 상속받음

        # 이미지가 미니매치 크기만큼 입력으로 들어옴
        # 입력과 출력을 제외한 중간 레이어의 뉴런 수는 128, 64로 설계
        # ReLU 활성화 함수를 사용, 한번에 dorpout 적용
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=128) # 3*32*32 -> 128
        self.fc2 = nn.Linear(in_features=128, out_features=64)      # 128 -> 64  
        self.dropout = nn.Dropout(0.5)                              # 50% 확률로 뉴런을 끔
        self.fc3 = nn.Linear(in_features=64, out_features=10)       # 64 -> 10


    def forward(self, x):   # forward propagation  
        x = torch.flatten(x, 1)    # 이미지를 1차원으로 펼침, 행렬 형태의 입력을 벡터로 변형 
        x = self.fc1(x)            # 첫번째 fully connected layer 통과
        x = F.relu(x)              # ReLU 활성화 함수 적용
        x = self.fc2(x)            # 두번째 fully connected layer 통과
        x = F.relu(x)              # ReLU 활성화 함수 적용
        x = self.dropout(x)        # dropout 적용
        x = self.fc3(x)            # 세번째 fully connected layer 통과
        return x
    

# Instance 생성
model = MLPnet()