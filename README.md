# PRISM

정치 도메인 멀티모달 팩트체킹 파이프라인. 클레임·포스트·OCR·이미지를 입력받아
`Supported / Refuted / NEI` 판정과 Toulmin 구조 보고서를 생성한다.
NEI는 버려지는 기권이 아니라 증거 불충분을 뜻하는 1급 판정으로 다룬다.

```
triage -> retrieve(E_T, E_I) -> S(m) 에 속한 검증기만 -> calibrate -> explain
```

## 설계

**Modality-Aware Triage** — 클레임의 모달리티 수요를 마스크 `m=(img, ent, time)`으로
분류하고, 그래프 레벨에서 전문 에이전트의 활성화 자체를 게이팅한다. 비활성 노드는
실행되지 않는다. 평균 4개 중 2.14개만 돌아 비용이 20% 줄어든다.

**Gated Evidence Retrieval** — 오픈 웹 대신 클레임에 연결된 신뢰 코퍼스로 검색을
제한한다(BM25, k₁=1.5, b=0.75, top-3). 성능의 최대 동인이다. 게이트를 제거하면
정확도가 zero-shot 수준으로 떨어지고 R→S 오류가 10.6% → 34.5%로 치솟는다.

**Deterministic-First Calibration** — 규칙을 먼저 적용하고 남는 경우만 LLM에 위임한다.

```
conflict(σ) = (ŷ_v=Refuted) ∧ (n_sup^txt ≥ 2) ∧ (n_ref^txt = 0) ∧ (E_I ≠ ∅)
ŷ = δ_θ(σ)  if δ_θ(σ) ≠ ⊥   else   ψ_guard(LLM(σ, {v_s}))
```

교차모달 충돌 시 `γ_v ≥ τ_v`면 시각 판정이 텍스트 합의를 뒤집고, `ψ_guard`가
Refuted → Supported 승격을 차단한다. `θ=(τ_v, τ_q)=(0.7, 3)`.
이 층이 NEI 기권 능력과 R→S 안전성을 만든다.

**Decoupled Toulmin Explanation** — 설명 생성기는 확정된 판정과 동결된 rationale만
받는다. 판정을 재계산하지 않으므로 설명의 환각이 분류를 오염시키는 경로가 구조적으로
막힌다. 판정 경로는 전부 temperature 0이고, 0.3은 설명 생성기에만 적용한다.

## 결과

백본을 GPT-4o-mini로 통제한 비교. Acc / macro-F1.

| | MOCHEG bin | MOCHEG 3-cls | RW-Post (evid) |
|---|---|---|---|
| Zero-shot | .633 / .611 | .531 / .502 | .562 / .477 |
| DEFAME | .730 / .714 | .554 / .528 | .465 / .451 |
| MiniCheck | .640 / .609 | .520 / .388 | .457 / .297 |
| RAGAR | — / .850 | — | — |
| **PRISM** | **.857 / .857** | **.613 / .549** | **.816 / .672** |

RAGAR 프로토콜(balanced 300)에서 F1 .857, RW-Post에서 DEFAME 대비 +22점.
클레임당 $0.005.

**Ablation.** Calibrator를 빼면 raw accuracy는 오르지만(.615 → .655) NEI 기권이
0.28 → 0으로 사라지고 R→S가 유의하게 증가한다(p<.01). 정확도를 일부 내주고 기권과
안전성을 얻는 의도된 트레이드오프다.

**설명 품질.** 구조적 완전성 1.00(자유 서술 0.17), Faithfulness 0.99,
Decision-Fidelity 1.00.

## 실행

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
echo "OPENAI_API_KEY=sk-..." > .env
```

```bash
# 단건
python main.py --claim "..." --image data/post.jpg --trace

# 배치 평가
python -m prism.evaluation.batch_eval --dataset mocheg --setting binary \
    --n-per-class 150 --output results/mocheg_bin.csv

# ablation
python -m prism.evaluation.batch_eval --dataset rwpost --data data/rwpost.json \
    --variant no_gating --output results/rwpost_closed.csv

# 단위 테스트 (네트워크·API 키 불필요)
pytest tests
```

`--variant` 는 `full / no_gating / no_triage / no_calibrator / no_orchestration`.
배치 평가는 한 건마다 append 하고, 재실행하면 완료된 `claim_id` 를 건너뛴다.

## 구조

```
prism/
  core/       state, graph, calibrator, llm_models, verdict
  agents/     triage, retrieval, verifiers, generator
  evaluation/ batch_eval, metrics, judge, dataset, baselines
scripts/      smoke.py   실제 API로 에이전트 수동 점검
tests/        네트워크를 타지 않는 단위 테스트
docs/SPEC.md  논문 표기와 코드의 대응
```

## 한계

교차모달 불일치에 기대는 구조라, 이미지-텍스트 정합성이 유지된 조작이나 단일 모달
편향에는 취약하다. 모든 모달리티가 같은 오답으로 수렴하면 calibration이 개입할 여지가
없다. 자원 제약으로 단일 실행 결과이며, 평가는 정치 도메인 두 벤치마크에 한정된다.
