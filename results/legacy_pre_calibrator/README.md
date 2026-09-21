# 이전 파이프라인 결과

캘리브레이터 도입 이전, 단순 다수결 집계로 돌린 실행 기록이다.
현재 `prism/evaluation/batch_eval.py` 가 쓰는 컬럼 구성과 다르므로 resume 대상이 아니다.

- `baseline_300.csv` — 보수적 집계. 300건 중 215건까지 진행
- `strict_debug_50.csv` — NEI 억제 프롬프트 50건 디버그. 같은 claim_id가 중복 기록되어 있다
  (고유 50건 / 전체 88행). 집계 시 `drop_duplicates(subset="claim_id", keep="last")` 필요
