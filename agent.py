import os
import ast
import json
import operator as op
import re
from collections import Counter

# ── 1. Phoenix OTEL 등록 (openai import 전에 반드시 먼저 실행) ──────────────
from phoenix.otel import register

PROJECT_NAME = "phoenix-demo-에이전트"

tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint="http://localhost:6006/v1/traces",
)

# ── 2. OpenAI 클라이언트 + 자동 계측 ─────────────────────────────────────────
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

OpenAIInstrumentor().instrument()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
)
MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")

# ── 3. 도구 구현 ──────────────────────────────────────────────────────────────

ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        fn = ALLOWED_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"허용되지 않는 연산자: {type(node.op).__name__}")
        return fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = ALLOWED_OPS.get(type(node.op))
        return fn(_eval_node(node.operand))
    raise ValueError(f"허용되지 않는 표현식: {type(node).__name__}")


def calculator(expression: str) -> str:
    """산술 표현식을 안전하게 계산합니다 (AST 화이트리스트 방식)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return json.dumps({"표현식": expression, "결과": result}, ensure_ascii=False)
    except ZeroDivisionError:
        return json.dumps({"오류": "0으로 나눌 수 없습니다"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"오류": str(e)}, ensure_ascii=False)


def text_analyzer(text: str) -> str:
    """한국어/영어 텍스트를 분석합니다 (어절 기반, 외부 라이브러리 불필요)."""
    sentences = [s.strip() for s in re.split(r"[.!?。！？]+", text) if s.strip()]
    words = text.split()
    word_count = len(words)
    avg_word_len = round(sum(len(w) for w in words) / word_count, 2) if words else 0

    clean = [re.sub(r"[^\w가-힣]", "", w).lower() for w in words]
    clean = [w for w in clean if len(w) > 1]
    top_words = [{"단어": w, "빈도": c} for w, c in Counter(clean).most_common(5)]

    result = {
        "단어_수": word_count,
        "문장_수": len(sentences),
        "평균_단어_길이": avg_word_len,
        "빈출_단어_Top5": top_words,
    }
    return json.dumps(result, ensure_ascii=False)


# ── 4. OpenAI function calling 스키마 ────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "산술 표현식을 계산합니다. 덧셈(+), 뺄셈(-), 곱셈(*), 나눗셈(/), 정수나눗셈(//), 거듭제곱(**), 나머지(%) 연산을 지원합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 산술 표현식 (예: '(3 + 5) * 2', '2 ** 10')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "한국어 또는 영어 텍스트를 분석하여 단어 수, 문장 수, 평균 단어 길이, 빈출 단어 Top 5를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "분석할 텍스트",
                    }
                },
                "required": ["text"],
            },
        },
    },
]

TOOL_MAP = {
    "calculator": calculator,
    "text_analyzer": text_analyzer,
}

# ── 5. 한국어 시스템 프롬프트 ─────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 유능한 한국어 AI 어시스턴트입니다.

사용 가능한 도구:
1. calculator  - 수학 계산이 필요할 때 사용하세요. 복잡한 산술도 정확하게 처리합니다.
2. text_analyzer - 텍스트 분석이 필요할 때 사용하세요. 단어 수, 문장 수, 빈출 단어를 분석합니다.

지침:
- 계산이나 텍스트 분석이 필요한 경우에만 도구를 사용하고, 그 외에는 직접 답변하세요.
- 답변은 항상 한국어로 제공하세요.
- 계산 결과는 구체적인 수치와 함께 명확하게 설명하세요.
- 텍스트 분석 결과는 사용자가 이해하기 쉽게 요약해서 전달하세요."""

# ── 6. 에이전트 루프 ──────────────────────────────────────────────────────────

MAX_TOOL_ROUNDS = 5


def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)

        for call in message.tool_calls:
            func_name = call.function.name
            func_args = json.loads(call.function.arguments)
            tool_fn = TOOL_MAP.get(func_name)
            if tool_fn is None:
                tool_result = json.dumps({"오류": f"알 수 없는 도구: {func_name}"}, ensure_ascii=False)
            else:
                tool_result = tool_fn(**func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": tool_result,
            })

    return "최대 도구 호출 횟수에 도달했습니다. 부분적인 결과를 확인하세요."


# ── 7. 데모 실행 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_queries = [
        "2의 10승에서 100을 뺀 값은 얼마인가요? 그리고 그 결과가 홀수인지 짝수인지 알려주세요.",
        "다음 텍스트를 분석해주세요: '인공지능은 현대 사회를 빠르게 변화시키고 있습니다. 많은 기업들이 AI를 도입하고 있으며, 이는 업무 효율을 크게 높이고 있습니다. 앞으로도 AI 기술은 계속 발전할 것입니다.'",
        "(123 + 456) * 7 을 계산해주세요.",
        "안녕하세요! 오늘 기분은 어떠세요?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"[질문 {i}] {query}")
        print("=" * 60)
        answer = run_agent(query)
        print(f"[답변]\n{answer}")

    print(f"\n\n모든 트레이스가 Phoenix에 저장되었습니다: http://localhost:6006")
    print(f"프로젝트 이름: {PROJECT_NAME}")
