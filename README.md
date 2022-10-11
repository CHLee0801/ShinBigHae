# 신.빅.해 (신한 빅데이터 해커톤 1회)

## 간단 요약

신빅회 1회 신한투자증권 부문 우승팀입니다.


주최 측에서 제공하는 데이터를 활용해 증권 성향 고객을 설정하고 

다양한 데이터 처리와 모델링을 통해 증권 성향 고객을 판단하고 잠재적 증권 성향 고객까지 유추할 수 있는 방법론을 제시하고자 합니다.

또한 이렇게 선정된 증권성향 고객의 특징을 분석하고, 향후 이렇게 선별된 고객 정보를 어떻게 활용할 수 있는지 제안합니다.

## 최종 발표 자료

![최종발표ppt.pdf](https://github.com/CHLee0801/ShinBigHae/files/9751241/ppt.pdf)

## 환경 설정
```
conda create -n shinbighae python=3.9 && conda activate shinbighae
pip install -r requirements.txt
```

## 모델 실행

주최 측에서 제공하는 데이터를 공개할 수 없기에 실행가능한 Source Code만 제공합니다.

> Data Analysis


데이터 분석에 활용된 코드입니다.

> DeepLearningModel


AutoEncoder, Bert에 대한 실행 코드입니다. Pytorch-Lightning 기반으로 구현하였습니다.