# SatelliteDet2025

- `configs/defm_detr_base.py`에 모든 경로는 상대 경로로 설정되어 있어서 아래와 같이 worksapce의 디렉토리 구조만 맞추면 경로를 수정할 필요가 없습니다.
```
├── dataset
│   └── soccer-players
├── RilabDetrBase
│   ├── configs
│   ├── datasets
│   ├── model
│   ├── pipeline
│   └── util
└── tblog
```
- 현재는 `train.py`가 작동하는지 확인하는 용도로 soccer-players 데이터셋만 있습니다.
