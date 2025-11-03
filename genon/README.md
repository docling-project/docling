## 프로젝트 구조

```
# 최상단은 docling 구조를 그대로 가져감
# doc-parser 관련은 genon 디렉터리에 작성
.
├── build-script # 빌드 위한 스크립트 파일 및 컨피그 파일
├── docling
├── docs
├── genon
│   ├── preprocessor # genos에서 실행 될 전처리기 이미지 및 facade 관련
│   │   ├── configs # gunicorn, supervisor 설정
│   │   ├── docker # 도커파일 위치
│   │   ├── env # 개발 시 설정 파일들
│   │   ├── facade # facade 코드 
│   │   │   ├── evaluation
│   │   │   │   └── test_files
│   │   │   │       ├── annotated
│   │   │   │       ├── pdf
│   │   │   │       └── result
│   │   │   ├── gitbook_doc
│   │   │   │   └── images
│   │   │   ├── legacy
│   │   │   └── legal_parser
│   │   │       ├── api
│   │   │       ├── commons
│   │   │       ├── parsers
│   │   │       ├── schemas
│   │   │       └── services
│   │   ├── scripts # 도커 이미지 push 및 디비 등록 관련 스크립트 위치
│   │   ├── resources # 폰트 및 기타 리소스 파일들
│   │   ├── sample_files
│   │   ├── src # 전처리기 API 소스
│   │   │   └── common
│   │   └── tests # genos doc-parser 테스트 소스
│   │       ├── regression
│   │       │   └── baselines
│   │       ├── smoke
│   │       └── unit
│   ├── serving # ocr 및 
│   │   └── paddle
│   │       ├── config # ocr, vl paddlex 실행 파일
│   │       ├── docker
│   │       ├── etc
│   │       ├── k8s-manifest
│   │       └── resources
│   └── tools
│       └── genos_tools
│           └── commands
└── tests # docling 리포 test 관련 코드 작성 x
```

## 전처리기 빌드 및 등록

1. build-script 디렉토리 이동
2. [doc-parser-build.config](../build-script/doc-parser-build.config) 변경 사항 반영
3. 실행 [doc-parser-build.sh](../build-script/doc-parser-build.sh)
4. [register.config](preprocessor/scripts/register.config) 변경 사항 있을 시 변경 필요
5. 실행 [register_image.sh](preprocessor/scripts/register_image.sh) : push와 디비에 등록해준다.
6. 사이트 배포 시
```shell
1. 이미지 저장
docker save mncregistry:30500/mnc/doc-parser-preprocessor:latest | gzip > doc-parser-preprocessor.tar.gz
2. 사이트에서 이미지 복원
gunzip -c doc-parser-preprocessor.tar.gz | docker load
3. register_image.sh 파일 실행
```

## paddle-ocr 빌드 및 배포

1. build-script 디렉토리 이동
2. [paddle-ocr-build.config](../build-script/paddle-ocr-build.config) 설정 파일 변경
3. [paddle-ocr-build.sh](../build-script/paddle-ocr-build.sh) 스크립트 실행 : 빌드 및 레지스트리 푸쉬
4. [doc-parser-ocr-deployment.yaml](serving/paddle/k8s-manifest/doc-parser-ocr-deployment.yaml)
```shell
kubectl apply -f doc-parser-ocr-deployment.yaml
```
5. 노드 포트로 배포시는 [doc-parser-ocr-deployment-node-port.yaml](serving/paddle/k8s-manifest/doc-parser-ocr-deployment-node-port.yaml)