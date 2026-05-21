import os
import pandas as pd
import phoenix as px
from phoenix.evals import OpenAIModel, llm_classify

# ── 1. Phoenix 클라이언트 및 평가 모델 설정 ───────────────────────────────────

PROJECT_NAME = "phoenix-demo-에이전트"

phoenix_client = px.Client(endpoint="http://localhost:6006")

eval_model = OpenAIModel(
    model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"),
    api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
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

다음 중 하나로만 답하세요 (다른 말은 절대 하지 마세요):
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

다음 중 하나로만 답하세요 (다른 말은 절대 하지 마세요):
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

다음 중 하나로만 답하세요 (다른 말은 절대 하지 마세요):
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

다음 중 하나로만 답하세요 (다른 말은 절대 하지 마세요):
간결
적당
장황"""

EVAL_CONFIGS = [
    {
        "name": "완결성",
        "template": COMPLETENESS_TEMPLATE,
        "labels": ["완전", "부분적", "불완전"],
    },
    {
        "name": "관련성",
        "template": RELEVANCE_TEMPLATE,
        "labels": ["관련", "부분관련", "무관"],
    },
    {
        "name": "유용성",
        "template": HELPFULNESS_TEMPLATE,
        "labels": ["매우유용", "유용", "보통", "유용하지않음"],
    },
    {
        "name": "간결성",
        "template": CONCISENESS_TEMPLATE,
        "labels": ["간결", "적당", "장황"],
    },
]

# ── 3. 트레이스 가져오기 ──────────────────────────────────────────────────────


def fetch_traces_as_dataframe() -> pd.DataFrame:
    """Phoenix에서 트레이스를 가져와 평가용 DataFrame으로 변환합니다."""
    spans_df = phoenix_client.get_spans_dataframe(project_name=PROJECT_NAME)

    if spans_df is None or spans_df.empty:
        print("경고: Phoenix에서 트레이스를 찾을 수 없습니다.")
        print("먼저 agent.py를 실행하여 트레이스를 생성하세요: python agent.py")
        return pd.DataFrame()

    # input/output 컬럼 확인
    input_col = next((c for c in spans_df.columns if "input" in c.lower() and "value" in c.lower()), None)
    output_col = next((c for c in spans_df.columns if "output" in c.lower() and "value" in c.lower()), None)

    if not input_col or not output_col:
        print(f"경고: 필요한 컬럼이 없습니다. 사용 가능한 컬럼:\n{list(spans_df.columns)}")
        return pd.DataFrame()

    eval_df = spans_df[spans_df[input_col].notna() & spans_df[output_col].notna()].copy()
    eval_df = eval_df.rename(columns={input_col: "input", output_col: "output"})

    print(f"평가할 스팬 수: {len(eval_df)}")
    return eval_df[["input", "output"]].reset_index(drop=True)


# ── 4. 평가 실행 ──────────────────────────────────────────────────────────────


def run_evaluations(eval_df: pd.DataFrame) -> dict:
    """4가지 평가 지표로 LLM-as-judge 평가를 실행합니다."""
    results = {}

    for config in EVAL_CONFIGS:
        print(f"\n[{config['name']}] 평가 실행 중...")
        result_df = llm_classify(
            dataframe=eval_df,
            template=config["template"],
            model=eval_model,
            rails=config["labels"],
            provide_explanation=True,
            concurrency=4,
        )
        results[config["name"]] = result_df
        label_counts = result_df["label"].value_counts().to_dict()
        print(f"  결과 분포: {label_counts}")

    return results


# ── 5. 결과 저장 및 Phoenix 업로드 ───────────────────────────────────────────


def save_results(eval_df: pd.DataFrame, results: dict) -> None:
    """평가 결과를 Phoenix에 업로드하고, 실패 시 CSV로 저장합니다."""
    upload_success = False

    for metric_name, result_df in results.items():
        try:
            phoenix_client.log_evaluations(
                evaluations=result_df,
                project_name=PROJECT_NAME,
                eval_name=metric_name,
            )
            print(f"[{metric_name}] Phoenix 업로드 완료")
            upload_success = True
        except Exception as e:
            print(f"[{metric_name}] Phoenix 업로드 실패: {e}")

    # CSV 폴백: 항상 로컬에도 저장
    all_results = pd.concat(
        [df.assign(지표=name, 질문=eval_df["input"].values, 답변=eval_df["output"].values)
         for name, df in results.items()],
        ignore_index=True,
    )
    output_path = "evaluation_results.csv"
    all_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n평가 결과 CSV 저장 완료: {output_path}")

    if not upload_success:
        print("※ Phoenix 업로드에 실패했습니다. CSV 파일을 Phoenix UI에서 직접 import하세요.")


# ── 6. 메인 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phoenix 트레이스 평가 시작")
    print(f"Phoenix 서버: http://localhost:6006")
    print(f"프로젝트: {PROJECT_NAME}")
    print("=" * 60)

    eval_df = fetch_traces_as_dataframe()

    if eval_df.empty:
        print("\n평가할 데이터가 없습니다. 먼저 에이전트를 실행하세요:")
        print("  python agent.py")
        raise SystemExit(1)

    results = run_evaluations(eval_df)
    save_results(eval_df, results)

    print("\n" + "=" * 60)
    print("평가 완료! Phoenix UI에서 결과를 확인하세요:")
    print("  http://localhost:6006")
