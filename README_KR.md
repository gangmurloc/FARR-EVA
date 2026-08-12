# FARR-EVA 연구 아티팩트

[![Tests](https://github.com/gangmurloc/FARR-EVA/actions/workflows/tests.yml/badge.svg)](https://github.com/gangmurloc/FARR-EVA/actions/workflows/tests.yml)

[English README](README.md)

이 저장소는 FLARE, IRCoT, FARR가 생성한 다중 홉 QA 후보를 검색 근거와
추론 trace에 기반해 사후 중재하는 FARR-EVA의 공개용 연구 코드입니다.
원래 실험 작업공간에서 핵심 코드, 작은 동결 selector, 테스트, 검증된
결과 요약만 분리했습니다.

## 현재 판정

- Test-C 6,000문항에서 FARR-EVA macro F1은 0.5754입니다.
- 고정 FARR 기준선 0.5140보다 +0.0614 높았고, 95% CI는
  [0.0529, 0.0700]입니다.
- 그러나 Test-C를 사후 재분석한 corrected portable selector는 0.5844로
  FARR-EVA보다 0.0090 높았습니다.
- 이 사후 비교는 확증 결과가 아니며, selector 비교를 판정하기 위한
  fresh Test-D 9,000문항은 결과가 나오기 전까지 공개하지 않습니다.

따라서 이 저장소는 FARR-EVA가 보편적으로 최고라고 주장하지 않습니다.
핵심 연구 질문, 구현, 잠금 프로토콜, 성공과 실패 진단을 재현 가능한
형태로 보여주는 포트폴리오용 연구 아티팩트입니다.

## 연구 질문

데이터셋 이름, expert identity, gold answer, gold supporting fact, runtime
counter 같은 직접적인 shortcut 정보를 추론 입력으로 사용하지 않고,
retrieved evidence와 reasoning trace에서 측정한 특성만으로 여러 QA
trajectory 중 더 신뢰할 수 있는 답변을 선택할 수 있는가?

## 작성자와 기여

**Ganggil Lee** — 한림대학교 NLP Laboratory 학부 연구생

- FARR-EVA evidence-vector 중재 구조 설계
- 후보 근거 측정과 selector 학습 구현
- 세 다중 홉 QA 데이터셋 평가 및 잠금 프로토콜 구성
- bootstrap 분석, 무결성 검사, 단위 테스트와 공개 아티팩트 정리

FLARE와 IRCoT는 후보 생성에 활용한 선행 방법이며, 이 저장소의 구현은
공식 재현본이 아닌 연구용 adaptation입니다. 자세한 경계와 출처는 영문
README의 Attribution 및 `THIRD_PARTY_NOTICES.md`에 명시했습니다.

## 실행

```bash
git clone https://github.com/gangmurloc/FARR-EVA.git
cd FARR-EVA
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
python examples/selector_demo.py
```

원본 데이터, 생성 결과, 대형 모델 weight, 로그, 논문 원고는
의도적으로 제외했습니다. 아직 오픈소스 라이선스를 부여하지 않았으므로,
공개 열람과 코드 재사용 허가는 동일하지 않습니다.
