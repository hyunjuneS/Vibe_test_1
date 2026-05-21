from pathlib import Path
import yaml

_path = Path(__file__).parent / "config.yml"

with open(_path, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

OPENAI_API_KEY: str = _cfg["openai"]["api_key"]
OPENAI_BASE_URL: str = _cfg["openai"]["base_url"]
OPENAI_MODEL_NAME: str = _cfg["openai"]["model_name"]

PHOENIX_HOST: str = _cfg["phoenix"]["host"]
PHOENIX_ENDPOINT: str = f"{PHOENIX_HOST}/v1/traces"
PROJECT_NAME: str = _cfg["phoenix"]["project_name"]
