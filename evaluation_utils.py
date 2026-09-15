"""Shared utilities for LLM-as-judge scoring (evaluate-metrics.py) and EFA result paths."""

import json
import re
from contextlib import nullcontext
from enum import Enum
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class Scale(Enum):
    BOOLEAN = "0 (No), 1 (Yes)"
    LIKERT = "1 (Very Poorly), 2 (Poorly), 3 (Adequately), 4 (Well), 5 (Very Well)"

    @property
    def bounds(self) -> tuple[int, int]:
        """Inclusive (min, max) valid score for this scale — used to reject out-of-range parses."""
        if self is Scale.BOOLEAN:
            return (0, 1)
        elif self is Scale.LIKERT:
            return (1, 5)
        else:
            raise ValueError(f"No bounds defined for scale {self!r}")


# Judge system prompt — SINGLE SOURCE OF TRUTH Every judge/scorer in this repo (the direct-API
# EFA builders AND the MLflow judges) uses this one preamble for the "## Role + ## Scoring
# Guidelines" block, so the role and scoring instructions can never drift between callers
# again.
SCORING_PREAMBLE = """## Role
You are an experienced K12 educator assessing a Tutor who is interacting with a Student.

## Scoring Guidelines
- Aim to be objective, unbiased, and constructive
- Assign low scores when warranted
- Focus solely on the responses and actions of the Tutor
- Your result must use the scale provided
- If a criterion does not apply to the conversation, assign it "N/A\""""


# ConvoLearn Dataset Loader

def parse_convolearn_turns(conversation_text: str) -> list[dict]:
    """Parse ConvoLearn conversation text into turns."""
    turns = []

    # Split on ' | ' delimiter
    segments = conversation_text.split(' | ')

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Parse role prefix
        if segment.startswith('System:'):
            # Skip system messages (question setup, "Moving to next question...")
            continue
        elif segment.startswith('Student'):
            # Handle "Student (Jamie):" or "Student:" variants
            match = re.match(r'Student(?:\s*\([^)]+\))?:\s*', segment)
            if match:
                content = segment[match.end():].strip()
                if content:
                    turns.append({'role': 'student', 'content': content})
        elif segment.startswith('Teacher:'):
            content = segment[8:].strip()  # len('Teacher:') = 8
            if content:
                turns.append({'role': 'tutor', 'content': content})

    return turns


def load_convolearn(
    csv_path: Path | str,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    """Load conversations from ConvoLearn CSV (full or split)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # offset then limit -> a deterministic disjoint slice, e.g. offset=250,limit=250 is
    # conversations 250-499.
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)

    conversations = {}
    metadata_rows = []

    for idx, row in df.iterrows():
        conv_id = row.get('conversation_id') or f"convolearn_{idx:05d}"
        turns = parse_convolearn_turns(row['conversationText'])
        if not turns:
            continue
        conversations[conv_id] = turns
        meta = row.to_dict()
        meta['conversation_id'] = conv_id
        metadata_rows.append(meta)

    return conversations, pd.DataFrame(metadata_rows)


@lru_cache(maxsize=4)
def _get_openai_client(base_url: str | None = None):
    """OpenAI-compatible client, reused across calls (cached per base_url)."""
    import openai
    if base_url:
        return openai.OpenAI(base_url=base_url, api_key="local", timeout=120.0, max_retries=0)
    return openai.OpenAI(timeout=120.0, max_retries=0)  # retries handled by tenacity wrapper below


def _transient_openai_errors():
    """Tuple of OpenAI exception types worth retrying (timeout, connection, rate limit, 5xx)."""
    import openai
    return (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
    )


def _is_reasoning_model(model_name: str) -> bool:
    """True for models that spend output tokens on hidden reasoning before the answer."""
    m = model_name.lower()
    if 'claude' in m or 'anthropic' in m:
        return True
    # Local open-weight judges (Gemma, Qwen, etc. served via Ollama/mlx-lm) emit
    # reasoning/preamble before the score, so a 50-token cap truncates the reply to empty
    # (finish_reason='length').
    if 'gemma' in m or 'qwen' in m or 'llama' in m or 'mistral' in m or 'phi' in m:
        return True
    return m.startswith('gpt-5') or m.startswith('o1') or m.startswith('o3') or m.startswith('o4')


def _max_completion_tokens_for(model_name: str) -> int:
    """Token budget for the judge's reply."""
    return 2000 if _is_reasoning_model(model_name) else 50


def model_name_for_filename(model: str) -> str:
    """Filesystem-safe model name for output paths: strips any provider prefix
    ('openai:/gpt-5.4-mini' -> 'gpt-5.4-mini') and the ':' a bare model id may carry."""
    return model.split('/')[-1].replace(':', '-')


def efa_result_dir(kind: str, model: str | None = None) -> Path:
    """Return (and create) the efa/results subfolder for a given artifact kind."""
    base = Path(__file__).parent / 'efa' / 'results'
    if kind == 'batch_inputs':
        if not model:
            raise ValueError("efa_result_dir('batch_inputs') requires a model for the subfolder")
        d = base / 'batch_inputs' / model_name_for_filename(model)
    else:
        d = base / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_provider(model: str) -> str:
    """Map a --model string to its provider: 'openai' | 'anthropic' | 'local'."""
    if model.startswith('anthropic'):
        return 'anthropic'
    if model.startswith(('local', 'ollama')):
        return 'local'
    return 'openai'


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_transient_openai_errors()),
    reraise=True,
)
def _create_completion_with_retry(model_name: str, messages: list[dict], base_url: str | None = None,
                                  response_format: dict | None = None):
    """Call chat.completions.create with exponential backoff on transient errors only."""
    max_output_tokens_options = "max_tokens" if base_url else "max_completion_tokens"
    max_output_tokens = {max_output_tokens_options: _max_completion_tokens_for(model_name)}
    # response_format=None is mapped to NOT_GIVEN by the SDK and dropped from the request, so
    # the default free-text call is unchanged; a schema (Structured Outputs) is sent only when
    # a caller passes one.
    return _get_openai_client(base_url).chat.completions.create(
        model=model_name,
        messages=messages,
        response_format=response_format,
        **max_output_tokens,
    )


def format_turns_as_text(turns: list[dict]) -> str:
    """Format conversation turns as plain text with STUDENT/TUTOR labels."""
    return "\n".join([
        f"{'STUDENT' if t['role'] in ('user', 'student') else 'TUTOR'}: {t['content']}"
        for t in turns
    ])


def build_messages(conversation_text: str, subcategory: str, prompt: str, scale: Scale) -> list[dict]:
    """Build the [system, user] messages for scoring one metric on a conversation."""
    system_content = f"""{SCORING_PREAMBLE}

## Conversation to Evaluate
{conversation_text}"""

    user_content = f"""Evaluate the conversation on the following criterion:
{prompt}

Rating scale: {scale.value}

Respond with ONLY the integer score or "N/A"."""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def parse_score_response(content: str | None, scale: Scale | None = None):
    """Parse a score from the judge's reply, validated against the scale range."""
    content = (content or '').strip().upper()

    # N/A must be the WHOLE reply (allow surrounding quotes/punctuation), not a passing
    # mention like "N/A is wrong, Score: 3" — a substring match would silently drop a real
    # score.
    if re.fullmatch(r'["\'\s]*N/?A["\'.\s]*', content):
        return None, 'not_applicable'

    # Prefer an explicit "Score: N" if present; else the FIRST standalone integer (the leading
    # score digit — see docstring).
    labeled = re.search(r'SCORE\s*[:=]?\s*(\d+)', content)
    standalone = re.search(r'\b\d+\b', content)
    picked = labeled.group(1) if labeled else (standalone.group() if standalone else None)
    if picked is None:
        return None, 'error'
    value = int(picked)
    if scale is not None:
        lo, hi = scale.bounds
        if value < lo or value > hi:
            return None, 'error'  # out-of-range — reject rather than clamp
    return value, 'scored'


def build_score_output_config(scale: Scale) -> dict:
    """Build Anthropic's `output_config` for a number-only judgment (Structured Outputs)."""
    lo, hi = scale.bounds
    allowed = list(range(lo, hi + 1))  # LIKERT -> [1..5]; BOOLEAN -> [0,1]
    return {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        # Mixed enum: the valid integers, plus the literal "N/A".
                        "enum": allowed + ["N/A"],
                        "description": f"The rubric score. One of {allowed}, or \"N/A\".",
                    },
                },
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }


def build_score_response_format(scale: Scale) -> dict:
    """Build OpenAI's `response_format` for a number-only judgment (Structured Outputs)."""
    lo, hi = scale.bounds
    allowed = list(range(lo, hi + 1))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rubric_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "anyOf": [
                            {"type": "integer", "enum": allowed},
                            {"type": "string", "enum": ["N/A"]},
                        ],
                        "description": f"The rubric score. One of {allowed}, or \"N/A\".",
                    },
                },
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }


def parse_score_json(text: str | None, scale: Scale | None = None):
    """Parse a Structured-Outputs reply ('{"score": 3}' or '{"score": "N/A"}') into the same
    (score, status) contract as parse_score_response."""
    if not text:
        return None, 'error'
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None, 'error'
    if not isinstance(obj, dict) or 'score' not in obj:
        return None, 'error'
    raw = obj['score']
    if isinstance(raw, str):
        if raw.strip().upper().replace('/', '') == 'NA':
            return None, 'not_applicable'
        try:
            raw = int(raw.strip())
        except ValueError:
            return None, 'error'
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, 'error'
    value = int(raw)
    if scale is not None:
        lo, hi = scale.bounds
        if value < lo or value > hi:
            return None, 'error'
    return value, 'scored'


def _serialize_response(response):
    """Convert any LLM API response into a JSON-serializable dict, provider-agnostic."""
    # 1. Pydantic v2 (OpenAI, Anthropic, LiteLLM ModelResponse all implement this)
    for method in ('model_dump',):
        fn = getattr(response, method, None)
        if callable(fn):
            try:
                return fn(), method
            except Exception:
                pass
    # 2. Pydantic model_dump_json (round-tripped to a dict)
    fn = getattr(response, 'model_dump_json', None)
    if callable(fn):
        try:
            return json.loads(fn()), 'model_dump_json'
        except Exception:
            pass
    # 3. Older SDK dict-style converters
    for method in ('to_dict', 'dict'):
        fn = getattr(response, method, None)
        if callable(fn):
            try:
                return fn(), method
            except Exception:
                pass
    # 4. Already a plain dict
    if isinstance(response, dict):
        return response, 'dict-passthrough'
    # 5. Plain object attribute bag
    try:
        return dict(vars(response)), 'vars'
    except Exception:
        pass
    # 6. Last resort — always succeeds, preserves raw text
    return str(response), 'str'


def _write_raw_record(raw_log_path, raw_log_lock, record: dict) -> None:
    """Append one JSON-Lines record for a single LLM call under a lock."""
    if raw_log_path is None:
        return
    try:
        line = json.dumps(record, default=str) + '\n'
        with (raw_log_lock or nullcontext()):
            with open(raw_log_path, 'a') as f:
                f.write(line)
    except Exception as e:
        print(f"  ⚠️  raw-log write failed: {type(e).__name__}: {e}")


def evaluate_single_metric(conv_id: str, turns: list[dict], metric_id: str,
                           category: str, subcategory: str, prompt: str,
                           model: str, scale: Scale, source: str = '',
                           raw_log_path=None, raw_log_lock=None,
                           base_url: str | None = None, structured: bool = False) -> dict:
    """Evaluate a single metric for a single conversation using direct OpenAI calls."""
    model_name = model.split('/')[-1]  # strip provider prefix (openai:/gpt-5.4-mini)

    conversation_text = format_turns_as_text(turns)
    messages = build_messages(conversation_text, subcategory, prompt, scale)

    # Shared identity + metric attributes — everything needed to rebuild the CSV row, plus
    # enough config to reproduce/audit the call.
    record = {
        'conversation_id': conv_id,
        'metric_id': metric_id,
        'source': source,
        'category': category,
        'subcategory': subcategory,
        'model': model_name,
        'scale': scale.value,
        'prompt': prompt,
    }

    response_format = build_score_response_format(scale) if structured else None

    try:
        response = _create_completion_with_retry(model_name, messages, base_url=base_url,
                                                 response_format=response_format)
        choice = response.choices[0]
        content = choice.message.content
        # Distinguish truncation (reasoning model ran out of budget before emitting an answer)
        # from genuinely unparseable output.
        if choice.finish_reason == 'length' and not (content or '').strip():
            score, status = None, 'truncated'
        elif structured:
            # Structured Outputs guarantees a JSON '{"score": ...}' payload.
            score, status = parse_score_json(content, scale)
        else:
            score, status = parse_score_response(content, scale)

    except Exception as e:
        print(f"  ❌ {conv_id[:8]}/{metric_id[:20]}: {type(e).__name__} (after retries)")
        _write_raw_record(raw_log_path, raw_log_lock, {
            **record,
            'score': None,
            'score_status': 'error',
            'response_provider': None,
            'response_serialization': None,
            'response': None,
            'error': f"{type(e).__name__}: {e}",
        })
        return {
            'conversation_id': conv_id,
            'metric_id': metric_id,
            'source': source,
            'category': category,
            'subcategory': subcategory,
            'score': None,
            'score_status': 'error'
        }

    payload, strategy = _serialize_response(response)
    _write_raw_record(raw_log_path, raw_log_lock, {
        **record,
        'score': score,
        'score_status': status,
        'response_provider': type(response).__module__,  # e.g. 'openai.types...' / 'anthropic.types...'
        'response_serialization': strategy,              # which serializer succeeded
        'response': payload,                             # full raw API object (usage, choices, ...)
        'error': None,
    })

    return {
        'conversation_id': conv_id,
        'metric_id': metric_id,
        'source': source,
        'category': category,
        'subcategory': subcategory,
        'score': score,
        'score_status': status
    }
