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

## 고정 기준선 전체 비교

FARR는 Test-C 결과를 본 뒤 고른 비교군이 아니라 validation에서 미리
고정한 anchor이자 primary comparator입니다. 또한 Test-C macro F1 기준으로
가장 강한 고정 단일 expert였습니다.

| 시스템 | Macro F1 | Macro EM | 분석상 역할 |
|---|---:|---:|---|
| RARR | 0.3971 | 0.3035 | 고정 단일 기준선 |
| RAG | 0.4227 | 0.3330 | 고정 단일 기준선 |
| FLARE | 0.5106 | 0.4088 | 고정 단일 기준선 |
| Embedded FLARE | 0.5109 | 0.4093 | 고정 후보 expert |
| IRCoT | 0.5121 | 0.4018 | 고정 후보 expert |
| **FARR** | **0.5140** | **0.4125** | **validation 고정 anchor 및 primary comparator** |
| **FARR-EVA** | **0.5754** | **0.4608** | **제안한 중재 계층** |

추가로 Test-C에서 데이터셋별 최강 고정 expert를 사후 선택한 보수적
진단 기준(FARR/FLARE/IRCoT 조합)과 비교해도 FARR-EVA는 macro F1이
+0.0468 높았고 95% CI는 [0.0391, 0.0547]이었습니다. 단, 이 기준의
expert identity는 Test-C 집계값으로 정했으므로 primary 확증 비교가 아닌
사후 진단입니다.

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

## 내부 패키지 이름

`farr_star` 네임스페이스에는 현재 FARR-EVA의 selector·근거 측정 코드와
이전 FARR-STAR, EPR, ODR 연구 모듈이 함께 포함되어 있습니다. 동결된
selector artifact가 `farr_star.eva_selector` 경로를 참조하므로 이 이름을
유지했습니다. 현재 FARR-EVA 결과에서 FARR-STAR를 별도 제안 방법으로
보고하지는 않습니다.

## 실행

```bash
git clone https://github.com/gangmurloc/FARR-EVA.git
cd FARR-EVA
python -m venv .venv
```

가상환경 활성화:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

설치 및 확인:

```bash
pip install -e .
python -m unittest discover -s tests -v
python examples/selector_demo.py
```

원본 데이터, 생성 결과, 대형 모델 weight, 로그, 논문 원고는
의도적으로 제외했습니다. 아직 오픈소스 라이선스를 부여하지 않았으므로,
공개 열람과 코드 재사용 허가는 동일하지 않습니다. 전체 소스의 출처와
호환성 검토가 끝난 뒤에만 프로젝트 전체 오픈소스 라이선스를 결정합니다.
