아래 문서에서 작성자(authors) 정보를 추출하라.

각 작성자는 다음 두 가지 정보를 포함한다:
- name: 작성자 이름
- affiliation: 소속 기관 또는 부서 (없으면 null)

반드시 아래 JSON 포맷으로만 반환하라:
{"authors": [{"name": "이름", "affiliation": "소속"}]}

<document>
{{raw_text}}
</document>         # json(default) | python
