import torch

class RandomSampler:
  def __init__(self, full_size, batch_size, last_drop=True):
    self.full_size = full_size    # dataset size 50000
    self.batch_size = batch_size  # mini batch size 64
    self.last_drop = last_drop
    self.shuffle()                # random index vector 생성
  
  def shuffle(self):
    self.index = torch.randperm(self.full_size)  # randompermutaion을 이용해 0~49999까지 랜덤한 순서로 섞음
    self.start = 0                # index 벡터의 시작
    self.end = self.batch_size    # index 벡터의 끝 (batch size)
    
    # start와 end값이 batch size만큼 증가하면서 index를 slicing해서 리턴
    
  def get_random_idx(self):       # 랜덤 index를 추출하는 메인 함수
    if self.end > len(self.index): 
      if self.last_drop:
        self.shuffle()
      elif self.start >= len(self.index):
        self.shuffle()
    
    # slicing 하고 start, end 값 업데이트
    idx = self.index[self.start:self.end] # index 뽑기: index 벡터의 start부터 end까지 slicing
    self.start += self.batch_size        # start값 batch size만큼 증가 (다음 batch로 넘어가기 위해)
    self.end += self.batch_size          
    return idx
  
  
  # 마지막 리턴되는 idx가 전체 사이즈보다 큰 값으로 리턴되는 것?
  # -> 파이썬에서는 슬라이싱구간을 설정할 때 뒤에 있는 값이 전체 값보다 커지는 경우에는
  #    내부적으로 전체값의 마지막까지만 슬라이싱하게 되어있음
  #    즉 알아서 미니배치보다 작은 값으로 데이터 배치가 생성될 것임