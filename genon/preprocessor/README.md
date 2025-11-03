uv sync --editable docling

## on boarding
### 1. GenOS
```shell
git clone https://github.com/mindsandcompany/GenOS.git .
```
### 2. git config 적용
```shell
git config --local push.recurseSubmodules check
git config --local status.submoduleSummary true
git config --local diff.submodule log
git config --local submodule.recurse true
git config --local alias.spull 'pull --recurse-submodules'
git config --local alias.supdate 'submodule update --remote --recursive'
git config --local alias.sclone 'clone --recurse-submodules'
git config --local alias.spush 'push --recurse-submodules=check'
```
### 3. submodule branch 설정 변경
```ini
# 브랜치 별 tracking할 doc_parser branch 지정
# .gitmodules GenOS project root에 위치
[submodule "container-services/preprocessors/docling-preprocessor/doc_parser"]
	path = container-services/preprocessors/docling-preprocessor/doc_parser
	url = https://github.com/mindsandcompany/doc_parser.git
	branch = develop
#branch 변경
```
```shell
### 설정 파일 변경 후
git submodule sync --recursive
```
### 4. submodule 받기
```shell
## 최초 설정 및 받아야할 때
git submodule update --init --recursive --remote
## OR 이미 설정 상태에서 업데이트 시
git submodule update --remote --recursive
## 상태 확인
git submodule status --recursive
```
---
### 이미지 빌드
```shell
## GenOS git directory root
vi script/llmops.config
## PREPROCESSOR VERSION 변경
PREPROCESSOR_TAG=2.41.0
### 빌드
cd script/build/build_container_services/
bash build_docling_preprocessor.sh
```
---

## ⚠️ 주의사항
1. **항상 서브모듈부터 푸시하기**
2. **서브모듈 detached HEAD 상태 방지**
    - 서브모듈을 수정해야 할 경우 서브모듈 디렉터리에서
   ```shell
    # develop 브랜치로 이동합니다.
    git checkout develop
    
    # (선택사항, 하지만 권장) 최신 코드를 받아옵니다.
    git pull
    ```
3. **커밋 메시지에 서브모듈 변경사항 명시**
4. **팀원들과 서브모듈 브랜치 협의**

### 서브모듈 상태 확인
```shell
git submodule status --recursive
```


### 서브모듈 리셋
```shell
git submodule deinit -f --all
git submodule update --init --recursive --remote
```


## 요약

**서브모듈 + 메인 프로젝트 동시 변경 시 순서**:

1. **서브모듈 먼저**: 커밋 → 푸시
2. **메인 프로젝트 나중**: 커밋 → 푸시
3. **Git 설정**: `push.recurseSubmodules=check`
4. **자동화 스크립트** 사용 권장
5. **pre-push hook**으로 안전장치 설정

이 순서를 지키지 않으면 서브모듈 참조가 깨져서 다른 개발자들이 문제를 겪을 수 있습니다.

