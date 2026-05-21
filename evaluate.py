import pandas as pd
from phoenix.client import Client
from phoenix.evals import ClassificationEvaluator, LLM, evaluate_dataframe

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME, PHOENIX_HOST, PROJECT_NAME

# ── 1. Phoenix 클라이언트 및 평가 LLM 설정 ────────────────────────────────────

phoenix_client = Client()

eval_llm = LLM(
    provider="openai",
    model=OPENAI_MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# ── 2. 한국어 평가 프롬프트 템플릿 ───────────────────────────────────────────

COMPLETENESS_TEMPLATE = """당신은 AI 답변의 완결성을 평가하는 전문 평가자입니다.

[사용자 질문]
{input}

[AI 답변]
{output}

평가 기준:
- 질문의 모든 측면을 다루고 있는가?
- 누락된 중요한 정보가 없는가?
- 필요한 계산이나 분석이 완전히 수행되었는가?

위 기준을 바탕으로 다음 중 하나로만 답하세요:
완전
부분적
불완전"""

RELEVANCE_TEMPLATE = """당신은 AI 답변의 관련성을 평가하는 전문 평가자입니다.

[사용자 질문]
{input}

[AI 답변]
{output}

평가 기준:
- 답변이 질문에 직접적으로 연관되어 있는가?
- 불필요하거나 관련 없는 내용이 포함되지 않았는가?
- 답변이 질문의 의도를 올바르게 파악하고 있는가?

위 기준을 바탕으로 다음 중 하나로만 답하세요:
관련
부분관련
무관"""

HELPFULNESS_TEMPLATE = """당신은 AI 답변의 유용성을 평가하는 전문 평가자입니다.

[사용자 질문]
{input}

[AI 답변]
{output}

평가 기준:
- 답변이 사용자의 실제 필요를 충족시키는가?
- 답변이 실행 가능하거나 이해하기 쉬운 정보를 제공하는가?
- 사용자가 이 답변을 통해 실질적인 도움을 받을 수 있는가?

위 기준을 바탕으로 다음 중 하나로만 답하세요:
매우유용
유용
보통
유용하지않음"""

CONCISENESS_TEMPLATE = """당신은 AI 답변의 간결성을 평가하는 전문 평가자입니다.

[사용자 질문]
{input}

[AI 답변]
{output}

평가 기준:
- 답변이 필요한 내용만 포함하고 있는가?
- 불필요한 반복이나 장황한 설명이 없는가?
- 핵심 내용이 명확하고 간결하게 전달되는가?

위 기준을 바탕으로 다음 중 하나로만 답하세요:
간결
적당
장황"""

# ── 3. 평가기 정의 ────────────────────────────────────────────────────────────

evaluators = [
    ClassificationEvaluator(
        name="완결성",
        llm=eval_llm,
        prompt_template=COMPLETENESS_TEMPLATE,
        choices={"완전": 1.0, "부분적": 0.5, "불완전": 0.0},
        include_explanation=True,
    ),
    ClassificationEvaluator(
        name="관련성",
        llm=eval_llm,
        prompt_template=RELEVANCE_TEMPLATE,
        choices={"관련": 1.0, "부분관련": 0.5, "무관": 0.0},
        include_explanation=True,
    ),
    ClassificationEvaluator(
        name="유용성",
        llm=eval_llm,
        prompt_template=HELPFULNESS_TEMPLATE,
        choices={"매우유용": 1.0, "유용": 0.75, "보통": 0.5, "유용하지않음": 0.0},
        include_explanation=True,
    ),
    ClassificationEvaluator(
        name="간결성",
        llm=eval_llm,
        prompt_template=CONCISENESS_TEMPLATE,
        choices={"간결": 1.0, "적당": 0.5, "장황": 0.0},
        include_explanation=True,
    ),
]

# ── 4. 트레이스 가져오기 ──────────────────────────────────────────────────────


def fetch_traces_as_dataframe() -> pd.DataFrame:
    """Phoenix에서 트레이스를 가져와 평가용 DataFrame으로 변환합니다."""
    spans_df = phoenix_client.get_spans_dataframe(project_name=PROJECT_NAME)

    if spans_df is None or spans_df.empty:
        print("경고: Phoenix에서 트레이스를 찾을 수 없습니다.")
        print("먼저 agent.py를 실행하여 트레이스를 생성하세요: python agent.py")
        return pd.DataFrame()

    input_col = next((c for c in spans_df.columns if "input" in c.lower() and "value" in c.lower()), None)
    output_col = next((c for c in spans_df.columns if "output" in c.lower() and "value" in c.lower()), None)

    if not input_col or not output_col:
        print(f"경고: 필요한 컬럼이 없습니다. 사용 가능한 컬럼:\n{list(spans_df.columns)}")
        return pd.DataFrame()

    eval_df = spans_df[spans_df[input_col].notna() & spans_df[output_col].notna()].copy()
    eval_df = eval_df.rename(columns={input_col: "input", output_col: "output"})

    print(f"평가할 스팬 수: {len(eval_df)}")
    return eval_df[["input", "output"]].reset_index(drop=True)


# ── 5. 평가 실행 ──────────────────────────────────────────────────────────────


def run_evaluations(eval_df: pd.DataFrame) -> pd.DataFrame:
    """4가지 평가기를 한 번에 실행하고 결과 DataFrame을 반환합니다."""
    print("\n평가 실행 중... (완결성 / 관련성 / 유용성 / 간결성)")
    results_df = evaluate_dataframe(
        dataframe=eval_df,
        evaluators=evaluators,
    )
    # 각 지표별 점수 요약 출력
    score_cols = [c for c in results_df.columns if c.endswith("_score")]
    for col in score_cols:
        name = col.replace("_score", "")
        mean_score = results_df[col].mean()
        print(f"  [{name}] 평균 점수: {mean_score:.2f}")
    return results_df


# ── 6. 결과 저장 및 Phoenix 업로드 ───────────────────────────────────────────


def save_results(eval_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """평가 결과를 Phoenix에 업로드하고, 항상 CSV로도 저장합니다."""
    try:
        phoenix_client.log_evaluations(
            evaluations=results_df,
            project_name=PROJECT_NAME,
        )
        print("\nPhoenix 업로드 완료")
    except Exception as e:
        print(f"\nPhoenix 업로드 실패: {e}")
        print("※ CSV 파일을 Phoenix UI에서 직접 import하세요.")

    output_path = "evaluation_results.csv"
    combined = pd.concat([eval_df, results_df], axis=1)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"평가 결과 CSV 저장 완료: {output_path}")


# ── 7. 메인 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phoenix 트레이스 평가 시작")
    print(f"Phoenix 서버: {PHOENIX_HOST}")
    print(f"프로젝트: {PROJECT_NAME}")
    print("=" * 60)

    eval_df = fetch_traces_as_dataframe()

    if eval_df.empty:
        print("\n평가할 데이터가 없습니다. 먼저 에이전트를 실행하세요:")
        print("  python agent.py")
        raise SystemExit(1)

    results_df = run_evaluations(eval_df)
    save_results(eval_df, results_df)

    print("\n" + "=" * 60)
    print("평가 완료! Phoenix UI에서 결과를 확인하세요:")
    print(f"  {PHOENIX_HOST}")
