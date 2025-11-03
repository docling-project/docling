## 의존성 관리

[pyproject.toml](pyproject.toml) 
```
# 아래 3개 의존성은 강제로 넣어놈. docling project구조를 그대로 따라가기 위해서
# doc-parser 코드를 따로 분리하는 과정에서 genon의 코드가 하위에 위치해 다음 의존성들은 
# 고정을 시킴.
# docling 2.41 기준 아래 버전
docling-core               2.42.0
docling-ibm-models         3.8.0
docling-parse              4.1.0
```