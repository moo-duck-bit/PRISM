"""검증기 프롬프트에 공통으로 들어가는 판정 지침.

NEI는 폐기되는 기권이 아니라 "증거 불충분"을 뜻하는 1급 판정이다.
따라서 검증기 단계에서 NEI를 억지로 억제하지 않는다. 기권 여부와 최종 수렴은
캘리브레이터(core/calibrator.py)가 검증기 합의와 교차모달 규칙으로 결정한다.

각 검증기는 자기 관할 밖의 사실관계를 판단하지 않는다. 관할을 좁혀 둬야
검증기 간 불일치가 실제 신호가 되고, 캘리브레이터가 그 신호로 판정을 교정할 수 있다.
"""

VERDICT_GUIDELINES = """Verdict guidelines:
1. Judge only within your own scope. Do not rule on parts of the claim that
   another verifier is responsible for.
2. Use only the evidence provided. Do not rely on knowledge that is absent from it.
3. Choose "Supported" or "Refuted" when the evidence bears on the claim, even if
   it does not match word for word. Reason from context.
4. Choose "NEI" when the evidence is insufficient or irrelevant to your scope.
   NEI is a valid verdict, not a failure to answer.
5. Set confidence to how strongly the evidence itself determines your verdict."""

RESPONSE_FORMAT = """Respond ONLY with valid JSON using these keys:
1. "verdict": exactly one of "Supported", "Refuted", "NEI".
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": one short sentence grounded in the supplied evidence.
   This sentence is carried into the final report verbatim, so it must not
   contain anything the evidence does not state."""

NO_EVIDENCE = "(No evidence was retrieved for this claim.)"
