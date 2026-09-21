# PRISM 설계 문서

논문 메소드와 코드의 대응을 정리한 문서다. 구현과 어긋나면 이 문서를 먼저 고친다.

## 0. 표기 대응

| 논문 | 코드 |
|---|---|
| `x = (c, P, O, I)` | `initial_state(claim, post_text, ocr_text, image_paths)` |
| `tau(c) -> m` | `agents/triage.py` |
| `S(m)` | `agents/triage.py::active_agents` |
| `E_T`, `E_I` | `agents/retrieval_text.py`, `agents/retrieval_visual.py` |
| `v_s = (y_s, gamma_s, r_s)` | `verdicts[s] = {verdict, confidence, rationale}` |
| `conflict(sigma)`, `delta_theta`, `psi_guard` | `core/calibrator.py` |
| `y_hat` | `state["verdict"]` |
| `E = g_Toulmin(c, {r_s}, y_hat)` | `agents/generator_toulmin.py` |

## 1. 파이프라인

```
triage -> retrieve_text(E_T) -> retrieve_visual(E_I)
       -> S(m) 에 속한 검증기만 -> calibrate -> explain
```

비활성 검증기는 프롬프트 안에서 건너뛰는 것이 아니라 **노드 자체가 실행되지 않는다.**
`core/graph.py` 의 조건부 엣지가 다음 활성 노드로 건너뛴다.

## 2. Triage (3.2절)

```
tau(c) = m = (m_img, m_ent, m_time)
S(m)   = {f} u {e | m_ent} u {st | m_time} u {v | m_img and |I| >= 1}
```

f는 항상 활성이다. v는 마스크가 켜져 있어도 실제 이미지가 없으면 활성화하지 않는다.
분류에 실패하면 `ent`, `time`을 켜는 쪽으로 기운다. 검증을 놓치는 것보다 낫다.

## 3. Gated Evidence Retrieval (3.3절)

```
E_T = TopK_{k=3}(BM25(D_c, c))        rank_bm25, k1=1.5, b=0.75
E_I = I  if m_img and |I| >= 1  else  empty
```

`D_c` 는 벤치마크가 정한 신뢰 증거 풀이다. 오픈 웹은 쓰지 않는다.

- MOCHEG: `claim_id` 로 연결된 Corpus3 문서 (문서 단위)
- RW-Post: 벤치마크 gold 증거 스니펫 (`Sample.evidence_pool` 로 전달)

연결 문서가 없는 클레임만 전체 코퍼스에서 폴백 검색하며, 이때 표본은 시드를 고정한다.
`gated=False` 는 게이트를 제거한 closed-book 경로로, `E_T` 가 비게 된다.

## 4. Deterministic-First Calibration (3.4절)

```
conflict(sigma) = (y_v = Refuted) and (n_sup_txt >= 2)
                  and (n_ref_txt = 0) and (E_I != empty)

y_hat = delta_theta(sigma)                  if delta_theta(sigma) != bottom
      = psi_guard(LLM(sigma, {v_s}))        otherwise
```

`theta = (tau_v, tau_q) = (0.7, 3)`.

**delta_theta 규칙 적용 순서** (안전성 우선)

1. `conflict(sigma)` 이면 `gamma_v >= tau_v` -> Refuted, 아니면 NEI
2. 정족수 `>= tau_q` 를 채운 라벨이 있으면 그 라벨
3. 교차모달 또는 모달리티 불일치면 NEI
4. 해당 없음이면 bottom, LLM에 위임

교차모달 충돌 규칙을 정족수보다 먼저 보는 이유는, 텍스트 검증기가 모두 Supported로
모여 정족수를 채운 상황이야말로 위험한 오탐이 발생하는 지점이기 때문이다.

**psi_guard**

1. 기권 가드: 충돌 상황에서 LLM이 Supported를 내면 시각 판정 또는 NEI로 교정한다.
   Refuted를 Supported로 올리는 승격은 허용하지 않는다.
2. 정족수 승격, 3. 과반 승격으로 LLM 출력을 검증기 합의에 맞춘다.

NEI 기권 능력은 주로 텍스트 우세 경로(2, 3)에서 나온다.
`use_calibrator=False` 는 1~3을 모두 끄고 LLM 출력을 그대로 쓴다. 이 경우 기권이 사라진다.

**결정성** — 판정 경로(triage, verification, vision, calibration)와 심판은 전부
temperature 0이다. `core/llm_models.py::TASK_SPECS` 한 곳에서 관리한다.

## 5. Decoupled Toulmin Explanation (3.5절)

```
E = g_Toulmin(c, {r_s}, y_hat)
```

생성기는 동결된 rationale과 확정된 `y_hat` 만 받는다. 원본 증거도, 판정 재계산도 없다.
설명 단계의 환각이 분류를 오염시키는 경로를 구조적으로 막는다.
출력은 6요소(claim, grounds, warrant, backing, qualifier, rebuttal) JSON으로 고정되고,
사람이 읽을 보고서는 LLM 없이 결정적으로 렌더링한다.
이 생성기만 temperature 0.3을 쓴다.

## 6. 평가

**지표** (`evaluation/metrics.py`)

| 지표 | 정의 |
|---|---|
| Acc, macro-F1 | 판정 품질. 균형 이진 세트에서는 weighted-F1과 같다 |
| R->S | `\|{y_hat=Sup \| y=Ref}\| / \|{y=Ref}\|`. 위험한 오탐 |
| NEI-rec | 기권 재현율 |
| Faithfulness | 설명의 원자 명제 중 `E_T u E_I u {r_s}` 가 함의하는 비율 |
| Decision-Fidelity | 설명의 결론이 정답이 아니라 `y_hat` 과 일치하는 비율 |

유의성은 paired McNemar(연속성 보정)와 부트스트랩 95% CI로 본다.

**ablation** (`--variant`)

| 이름 | 끄는 것 |
|---|---|
| `full` | - |
| `no_gating` | 증거 게이팅 (closed-book) |
| `no_triage` | 모달리티 게이팅. 4개 에이전트 전부 실행 |
| `no_calibrator` | delta_theta, psi_guard |
| `no_orchestration` | 위 둘 다 |

**실행**

```bash
# MOCHEG 균형 이진 300
python -m prism.evaluation.batch_eval --dataset mocheg --setting binary \
    --n-per-class 150 --output results/mocheg_bin.csv

# MOCHEG 3-class
python -m prism.evaluation.batch_eval --dataset mocheg --setting three_class \
    --output results/mocheg_3cls.csv

# RW-Post, 증거 게이팅 제거
python -m prism.evaluation.batch_eval --dataset rwpost --data data/rwpost.json \
    --variant no_gating --output results/rwpost_closed.csv

# 베이스라인
python -m prism.evaluation.batch_eval --dataset mocheg --baseline zero_shot \
    --output results/b2.csv
```

한 건마다 append 하고, 재실행하면 완료된 `claim_id` 를 건너뛴다.
샘플 순서는 섞지 않는다. resume이 순서에 기대기 때문이다.

**베이스라인** — B1(Majority), B2(Zero-shot)은 `evaluation/baselines.py` 에 있다.
B3(DEFAME), B4(MiniCheck)는 외부 시스템이라 구현하지 않고,
`predict_from_file` 로 예측 CSV를 읽어 같은 지표 계산에 태운다.
비교 시 백본은 GPT-4o-mini(temperature 0)로 통제하고, DEFAME의 오픈 웹 검색은
PRISM과 같은 증거 풀로 교체한다.

**정답 누출 방지** — RW-Post의 `reasoning_logic`, `key_points` 등 fact-checker 결론
필드는 읽지 않는다. `evaluation/dataset.py::LEAKING_FIELDS` 에 명시되어 있고
테스트가 강제한다.

## 7. 테스트

`tests/` 는 네트워크를 타지 않는다. 캘리브레이션 규칙, S(m) 계산, 그래프 라우팅,
BM25 순위, 지표, 데이터 로딩처럼 결정적으로 검증 가능한 부분만 다룬다.
모델 응답을 눈으로 볼 때는 `scripts/smoke.py` 를 쓴다.

## 8. 남은 작업

- 교차모달 정합이 유지된 조작(SAMM)과 단일 모달 편향(VERITE)은 불일치 신호가 없어
  현재 캘리브레이션이 개입할 여지가 없다.
- 텍스트 검증 3종과 시각 검증은 서로 독립이므로 fan-out / fan-in 병렬화 여지가 있다.
- 단일 실행 결과이며 multi-seed 평가는 미수행이다.
