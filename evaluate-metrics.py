#!/usr/bin/env python3
"""Score each conversation on each metric independently with an LLM judge."""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_utils import (
    Scale,
    build_messages,
    build_score_output_config,
    build_score_response_format,
    evaluate_single_metric,
    format_turns_as_text,
    load_convolearn,
    parse_score_response,
    parse_score_json,
    model_name_for_filename,
    resolve_provider,
    efa_result_dir,
    _max_completion_tokens_for,
)


# DATA LOADING & METRIC PREP

def print_status_summary(results_df, show_metrics: bool = False) -> None:
    """Print the scored/N-A/error/truncated breakdown for a results DataFrame."""
    counts = results_df['score_status'].value_counts()
    n = lambda s: int(counts.get(s, 0))
    print(f"  Scored: {n('scored')} | N/A: {n('not_applicable')} | "
          f"Errors: {n('error')} | Truncated: {n('truncated')}")
    if show_metrics:
        print(f"  Conversations: {results_df['conversation_id'].nunique()} | "
              f"Metrics: {results_df['metric_id'].nunique()}")
    if n('truncated') > 0:
        print(f"  ⚠️  {n('truncated')} truncated — raise max_completion_tokens for this model "
              f"(see _max_completion_tokens_for)")


def extract_prompt_keyword(prompt: str, max_words: int = 3) -> str:
    """Extract a short keyword/phrase from a prompt for disambiguation."""
    if not prompt or pd.isna(prompt):
        return "unknown"

    prompt = str(prompt).strip().rstrip('?.')

    # Remove common prefixes
    for prefix in ['The tutor ', 'Does the tutor ', 'Does the AI ', 'The AI ',
                   'Overall, the tutor ', 'Overall, ', 'To what extent ',
                   'was this tutor ', 'Criterion checks ']:
        if prompt.lower().startswith(prefix.lower()):
            prompt = prompt[len(prefix):]
            break

    words = prompt.split()[:max_words]
    keyword = '_'.join(words).lower()
    keyword = re.sub(r'[^a-z0-9_]', '', keyword)
    return keyword[:30] if keyword else "unknown"


def generate_metric_ids(df: pd.DataFrame) -> pd.Series:
    """Generate unique metric IDs from Category, Subcategory, and Prompt."""
    ids = []
    seen = {}

    for _, row in df.iterrows():
        cat = str(row.get('Category', '') or 'unknown').replace(' ', '_')
        cat = re.sub(r'[^a-zA-Z0-9_]', '', cat)

        sub = str(row.get('Subcategory', '') or 'unknown').replace(' ', '_').replace('-', '_')
        sub = re.sub(r'[^a-zA-Z0-9_]', '', sub)

        base_id = f"{cat}_{sub}"

        # Track prompts per base_id to detect duplicates
        prompt = row.get('Prompt', '')
        if base_id not in seen:
            seen[base_id] = {}
        if prompt not in seen[base_id]:
            seen[base_id][prompt] = len(seen[base_id])

        # Add keyword suffix if there are duplicates
        if len(seen[base_id]) > 1 or seen[base_id][prompt] > 0:
            keyword = extract_prompt_keyword(prompt)
            metric_id = f"{base_id}_{keyword}"
        else:
            metric_id = base_id

        ids.append(metric_id)

    # Second pass: handle any remaining collisions
    final_ids = []
    id_counts = {}
    for metric_id in ids:
        if metric_id in id_counts:
            id_counts[metric_id] += 1
            final_ids.append(f"{metric_id}_{id_counts[metric_id]}")
        else:
            id_counts[metric_id] = 0
            final_ids.append(metric_id)

    return pd.Series(final_ids, index=df.index)


def extract_tutor_output(content: str) -> str:
    """Extract only the <output-cai> content from tutor messages, stripping tool artifacts."""
    pattern = r'<output-cai>(.*?)</output-cai>'
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        return '\n\n'.join(matches).strip()
    # If no output-cai tags, return original (older format)
    return content


def load_conversations_from_json(input_dir: Path, limit: int | None = None, strip_tool_output: bool = True, offset: int = 0) -> dict[str, list[dict]]:
    """Load conversations from JSON files."""
    conversations = {}
    files = sorted(input_dir.glob("*.json"))
    if offset:
        files = files[offset:]
    if limit:
        files = files[:limit]

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        conv_id = f.stem  # Use filename as ID for consistency with token cost analysis

        turns = []
        for t in data.get("turns", []):
            content = t.get("content", "")
            if strip_tool_output and t.get("role") == "tutor":
                content = extract_tutor_output(content)
            turns.append({"role": t.get("role"), "content": content})

        if turns:
            conversations[conv_id] = turns

    return conversations


def load_conversations(input_path: Path, limit: int | None = None, strip_tool_output: bool = True, metadata_output_path: Path | None = None, offset: int = 0) -> dict[str, list[dict]]:
    """Load conversations from JSON directory or CSV file."""
    if input_path.is_file() and input_path.suffix == '.csv':
        conversations, metadata_df = load_convolearn(input_path, limit=limit, offset=offset)
        if metadata_output_path:
            metadata_df.to_csv(metadata_output_path, index=False)
        print(f"Loaded {len(conversations)} conversations from CSV")
        return conversations
    else:
        conversations = load_conversations_from_json(input_path, limit, strip_tool_output, offset=offset)
        print(f"Loaded {len(conversations)} conversations from JSON")
        return conversations


def load_metrics(metrics_file: Path) -> pd.DataFrame:
    """Load and normalize metrics CSV (supports both MetricCategories and Metrics formats)."""
    df = pd.read_csv(metrics_file)

    if 'Metric code' not in df.columns:
        # MetricCategories format - no special handling needed
        return df[['Category', 'Subcategory', 'Prompt']]

    # SOURCE 1: Jurenka et al. 2024 – LearnLM Responsible AI for Education Skip pairwise
    # comparison dimensions
    is_jurenka = df['Source'].isin(['(Jurenka et al. 2024) – LearnLM Responsible AI for Education', '(Jurenka et al., 2024) – LearnLM Responsible AI for Education'])
    df = df[~(is_jurenka & df['Metric dimension'].isin([
        'Pedagogy',  # "Which conversation exemplifies better tutor behavior..."
        'Accuracy',  # "Which conversation is better in terms of the accuracy..."
        'Human-likeness',  # "In which conversation was the tutor most like an excellent human tutor?"
        'Understand',  # "In which conversation did the tutor seem to better understand the student?"
        'Help',  # "In which conversation did the tutor better help the student?"
    ]))]
    # Skip metric with incompatible output format (expects binary judgment instead of 1-5 scale)
    df = df[~(is_jurenka & (df['Metric dimension'] == 'Guide towards the answer - checking that the tutor reveals the answer'))]

    # SOURCE 2: Maurya 2025 – MRBench Keep all metrics

    # SOURCE 3: Khoo 2025 – MinorBench Skip entire source - harm-topic gates scored N/A in
    # ~all conversations (or near-constant at ceiling), so no usable variance for factor
    # analysis.
    df = df[df['Source'] != '(Khoo et al., 2025) – MinorBench']

    # SOURCE 4: Macina et al., 2025 – MathTutorBench Skip entire source - task-specific
    # prompts not generalizable to tutor evaluation
    df = df[df['Source'] != '(Macina et al., 2025) – MathTutorBench']

    # SOURCE 5: Otero et al., 2025 (Nancy) Skip entire source - algebra misconception
    # detection, not tutor quality
    df = df[~df['Source'].isin(['(Otero et al., 2024)', '(Otero et al., 2025)'])]

    # SOURCE 6: Miller & DiCerbo, 2024 Skip entire source - student response correctness
    # evaluation, not tutor quality
    df = df[df['Source'] != '(Miller & DiCerbo, 2024)']

    # SOURCE 7: LearnLM 2024 – Improving Gemini for Learning
    is_learnlm_2024 = df['Source'] == '(LearnLM, 2024) – Improving Gemini for Learning'
    # For rows with NaN Metric dimension, use Prompt as Subcategory (single-tutor eval questions)
    df.loc[is_learnlm_2024 & df['Metric dimension'].isna(), 'Metric dimension'] = \
        df.loc[is_learnlm_2024 & df['Metric dimension'].isna(), 'Metric / Prompt definition']
    # Skip open-ended user survey (not scoreable by LLM)
    df = df[~(is_learnlm_2024 & df['Metric / Prompt definition'].str.contains(
        'Briefly, what was your impression of this tutor', case=False, na=False))]
    # Skip pairwise comparison questions
    df = df[~(is_learnlm_2024 & df['Metric dimension'].isin([
        'Which tutor did you prefer?',
        'Optionally, can you explain your preference?',
        'In which conversation were you better able to achieve your "learning goal"?',
        'Which tutor better adapted to your needs and proficiency as a student?',
        'Which conversation was an overall better experience?',
        'Feel free to share any other feedback on your experience with these two tutors.',
    ]))]

    # SOURCE 8: LearnLM 2025
    is_learnlm_2025 = df['Source'] == '(LearnLM, 2025)'
    # Skip pairwise/ranking metrics (Metric dimension is NaN, filter by Prompt)
    df = df[~(is_learnlm_2025 & df['Metric / Prompt definition'].isin([
        'Direct win rate',  # Pairwise win rate metric
        'ELO (Bradley-Terry model)',  # ELO ranking metric
    ]))]

    # SOURCE 9: TutorBench 2025
    is_tutorbench = df['Source'].isin(['(Srinivasa, et al., 2025) – TutorBench', '(Srinivasa et al., 2025) – TutorBench'])
    # Skip problem-solving generation prompt (not evaluation)
    df = df[~(is_tutorbench & df['Metric / Prompt definition'].str.contains(
        'You are a helpful math tutor. Solve the question step-by-step', case=False, na=False))]
    # Skip visual/image metrics (OUT OF SCOPE - require image input)
    df = df[~(is_tutorbench & df['Metric dimension'].isin([
        'Visual Perception',
        'Visual Reasoning',
    ]))]
    # Skip the 8 "high-level tutoring skill" tags.
    df = df[~(is_tutorbench & df['Metric / Prompt definition'].isin([
        'Stating definitions/formulae/theorems/laws',
        'Identifying correct steps by student',
        'Identifying incorrect steps by student',
        'Identifying core difficulty/misconception attribution',
        'Step by step help/analysis',
        'Asks questions to guide students',
        'Includes examples/analogy',
        'Provides alternative solutions/paths',
    ]))]

    # SOURCE 10: ConvoLearn Keep all metrics

    # SOURCE 11: EduBench Keep all metrics

    # SOURCE 12: OpenLearnLM
    is_openlearnlm = df['Source'] == '(Lee et al., 2026) – OpenLearnLM'
    # Skip dataset descriptions (not evaluation prompts)
    df = df[~(is_openlearnlm & df['Metric / Prompt definition'].isin([
        '918 MCQs (798 from C-Eval, 120 from GPQA)',  # Content Knowledge MCQ count
        '1,386 MCQs (243 from KICE, 1,143 from Pedagogy Benchmark).',  # Pedagogical Knowledge MCQ count
    ]))]
    # Skip dataset reference (not evaluation prompt)
    df = df[~(is_openlearnlm & df['Metric / Prompt definition'].isin([
        'See sub-scenarios dataset',
    ]))]
    # Skip meta-questions about model behavior (not tutor quality)
    df = df[~(is_openlearnlm & df['Metric dimension'].isin([
        'Behavioral Consistency',  # "Is the model's response strategy consistent across monitored (A) and unmonitored (B) conditions?"
    ]))]

    # SOURCE 13: EduDial Keep all metrics

    # SOURCE 14: EducationQ Keep all metrics (full definitions now in CSV)

    # SOURCE 15: TEACH-AI
    is_teachai = df['Source'] == '(Ding & Magerko, 2025) – TEACH-AI'
    # Skip meta questions about system (not tutor quality)
    df = df[~(is_teachai & df['Metric / Prompt definition'].isin([
        'How would you verify the explanation?',  # Explainability - meta question
        'How quickly does the AI provide adaptive feedback (e.g., latency)?',  # Adaptivity - requires timing data
        'Does the system behave consistently across different conditions, prompts, or contexts?',  # Consistency - requires multiple sessions
        'Was the AI easy to use and navigate?',  # System Usability - meta question
        'Has a safety or privacy audit been conducted?',  # Responsibility and Ethics - meta question
        'Were roles and responsibilities clear throughout the collaboration process?',  # Workflow Integration - about collaboration, not tutor quality
    ]))]

    # SOURCE 16: KMP-Bench
    is_kmp = df['Source'] == '(Shi et al., 2026) – KMP-Bench'
    # Skip accuracy/ranking metrics (in CSV order)
    df = df[~(is_kmp & df['Metric dimension'].isin([
        'Holistic win/lose/tie',  # Overall Judgement Acc. (from evaluator's holistic Win/Tie/Lose decision)
        'Average of General-Level Accuracy and mean of Principle-Level Accuracies',  # Composite Overall Acc. (average of General-Level Acc. and mean of six Principle-Level Accs.)
        'Win rate',  # General-Level Acc. (average win rate across 4 general criteria); Principle-Level Acc. (average win rate of 3 specific criteria for each of 6 pedagogical principles)
    ]))]
    # Skip General Criteria (pairwise comparison prompts)
    df = df[~(is_kmp & df['Metric dimension'].isin([
        'Linguistic Quality',  # NOT PUBLICLY AVAILABLE
        'Contextual Coherence and Relevance',  # pairwise comparison prompt
        'Adherence to persona and teaching style',  # pairwise comparison prompt
        'Adherence to specified principles',  # NOT PUBLICLY AVAILABLE
    ]))]
    # Transform: Pedagogical Principle Criteria → use Higher order category as Subcategory, Category definition as Prompt
    is_kmp = df['Source'] == '(Shi et al., 2026) – KMP-Bench'  # Recompute after filtering
    is_kmp_pedagogical = is_kmp & df['Higher order category'].str.contains('Pedagogical Principle Criteria', na=False)
    df.loc[is_kmp_pedagogical, 'Metric dimension'] = df.loc[is_kmp_pedagogical, 'Higher order category']
    df.loc[is_kmp_pedagogical, 'Metric / Prompt definition'] = df.loc[is_kmp_pedagogical, 'Category definition']

    # FINAL: Rename columns and drop incomplete rows
    df = df.rename(columns={
        'Metric code': 'Category',
        'Metric dimension': 'Subcategory',
        'Metric / Prompt definition': 'Prompt'
    })
    # A metric is valid as long as it has a Prompt (the definition the judge scores) and a
    # Category (for grouping).
    df = df.dropna(subset=['Category', 'Prompt'])

    # Remove duplicate (Category, Subcategory, Prompt) tuples
    df = df.drop_duplicates(subset=['Category', 'Subcategory', 'Prompt'])

    df = df[['Source', 'Category', 'Subcategory', 'Prompt']].reset_index(drop=True)
    df['metric_id'] = generate_metric_ids(df)

    return df


# BATCH TRANSPORT — OpenAI (OpenAI Batch API - 50% cost reduction)

def create_batch_file(tasks: list[tuple], scale: Scale, model: str, output_dir: Path,
                      offset: int = 0, limit: int | None = None,
                      structured: bool = False) -> Path:
    """Create JSONL file for OpenAI Batch API."""
    from datetime import datetime
    from collections import defaultdict

    model_name = model_name_for_filename(model)
    # Structured Outputs schema (openai:/ and local:/), built once — same for every request.
    response_format = build_score_response_format(scale) if structured else None
    # Name by chunk (offset/limit) AND timestamp: the chunk suffix makes the 4 chunks of one
    # run distinct and human-readable; the timestamp distinguishes separate runs (e.g. a re-
    # run on another day) so they don't overwrite each other.
    chunk_suffix = f"_o{offset}_l{limit}" if (offset or limit) else ""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_file = output_dir / f"batch_input_{model_name}{chunk_suffix}_{timestamp}.jsonl"

    # Group tasks by conversation_id to maximize cache hits
    tasks_by_conv = defaultdict(list)
    for cid, turns, metric_row in tasks:
        tasks_by_conv[cid].append((turns, metric_row))

    with open(batch_file, 'w') as f:
        for cid, conv_tasks in tasks_by_conv.items():
            turns = conv_tasks[0][0]  # All tasks for same conv have same turns
            conversation_text = format_turns_as_text(turns)  # format once per conversation
            for _, metric_row in conv_tasks:
                messages = build_messages(
                    conversation_text,
                    metric_row['Subcategory'],
                    metric_row['Prompt'],
                    scale,
                )

                body = {
                    "model": model_name,
                    "messages": messages,
                    "max_completion_tokens": _max_completion_tokens_for(model_name),
                }
                # Structured Outputs only when requested; free-text body is left unchanged.
                if response_format is not None:
                    body["response_format"] = response_format
                request = {
                    "custom_id": f"{cid}|{metric_row['metric_id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(request) + '\n')

    return batch_file


def submit_batch(batch_file: Path) -> str:
    """Upload batch file and submit job to OpenAI. Returns batch ID."""
    import openai
    client = openai.OpenAI()

    # Upload the JSONL file
    with open(batch_file, 'rb') as f:
        file_obj = client.files.create(file=f, purpose='batch')

    # Create the batch job
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch.id


def check_batch_status(batch_id: str) -> dict:
    """Check status of a batch job."""
    import openai
    client = openai.OpenAI()
    batch = client.batches.retrieve(batch_id)

    return {
        'id': batch.id,
        'status': batch.status,
        'created_at': batch.created_at,
        'completed_at': batch.completed_at,
        'failed_at': batch.failed_at,
        'request_counts': {
            'total': batch.request_counts.total,
            'completed': batch.request_counts.completed,
            'failed': batch.request_counts.failed
        },
        'output_file_id': batch.output_file_id,
        'error_file_id': batch.error_file_id
    }


def retrieve_batch_results(batch_id: str, metrics: pd.DataFrame, output_dir: Path,
                           scale: Scale = None) -> pd.DataFrame:
    """Download batch results and convert to evaluation CSV."""
    from datetime import datetime
    import openai
    client = openai.OpenAI()

    batch = client.batches.retrieve(batch_id)

    # Accept 'completed' and 'expired'.
    if batch.status not in ('completed', 'expired'):
        raise ValueError(f"Batch not retrievable. Status: {batch.status} "
                         f"(wait for 'completed', or 'expired' for partial results)")
    if batch.status == 'expired':
        print(f"  ⚠️  Batch EXPIRED — retrieving partial results only. "
              f"Completed {batch.request_counts.completed}/{batch.request_counts.total}.")

    if not batch.output_file_id:
        raise ValueError("No output file available (no requests completed)")

    # Download results file
    content = client.files.content(batch.output_file_id)
    results_text = content.text

    # Also surface failed requests: OpenAI writes errored/expired requests to a SEPARATE error
    # file that the output file never includes.
    n_errored = 0
    if batch.error_file_id:
        err_text = client.files.content(batch.error_file_id).text
        n_errored = sum(1 for ln in err_text.strip().split('\n') if ln.strip())
        print(f"  ⚠️  {n_errored} requests in the error file (not scored). "
              f"First few:")
        for ln in [l for l in err_text.strip().split('\n') if l.strip()][:3]:
            print(f"      {ln[:160]}")

    # Build metric lookup for category/subcategory info
    metric_lookup = {row['metric_id']: row for _, row in metrics.iterrows()}

    raw_records = []

    # Parse each result line
    all_results = []
    for line in results_text.strip().split('\n'):
        result = json.loads(line)
        custom_id = result['custom_id']
        conv_id, metric_id = custom_id.split('|', 1)

        metric_row = metric_lookup.get(metric_id, {})

        error = result.get('error')
        response = result.get('response', {})

        # Parse using the SAME logic as the sync path (scale-range validation + truncation
        # detection) so batch and sync produce identical scoring.
        if error:
            score, status = None, 'error'
        else:
            body = response.get('body', {})
            choices = body.get('choices', [])
            if choices:
                choice = choices[0]
                content = choice.get('message', {}).get('content')
                if choice.get('finish_reason') == 'length' and not (content or '').strip():
                    score, status = None, 'truncated'
                else:
                    score, status = parse_score_json(content, scale)
                    if status == 'error':
                        score, status = parse_score_response(content, scale)
            else:
                score, status = None, 'error'

        all_results.append({
            'conversation_id': conv_id,
            'metric_id': metric_id,
            'category': metric_row.get('Category'),
            'subcategory': metric_row.get('Subcategory'),
            'score': score,
            'score_status': status
        })

        # Raw record mirrors the sync path's schema.
        raw_records.append({
            'conversation_id': conv_id,
            'metric_id': metric_id,
            'source': metric_row.get('Source', ''),
            'category': metric_row.get('Category'),
            'subcategory': metric_row.get('Subcategory'),
            'model': response.get('body', {}).get('model'),
            'scale': scale.value if scale else None,
            'prompt': metric_row.get('Prompt'),
            'score': score,
            'score_status': status,
            'response_provider': 'openai-batch',
            'response_serialization': 'batch-passthrough',
            'response': result,
            'error': error if error else None,
        })

    # Name the output from the model that actually scored the batch (read from the results),
    # the batch id, and a timestamp — not from a caller-supplied --model.
    model_name = model_name_for_filename(raw_records[0]['model']) if raw_records else 'unknown'
    bid = batch_id.replace('batch_', '')[:12]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"efa_evaluation_results_{model_name}_batch_{bid}_{timestamp}.csv"
    raw_log_path = output_path.with_name(output_path.stem + '_raw.jsonl')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    with open(raw_log_path, 'w') as f:
        for rec in raw_records:
            f.write(json.dumps(rec, default=str) + '\n')
    print(f"  Raw LLM responses logged to {raw_log_path.name}")

    # Loud completeness check: batch.request_counts.total is what was submitted; if the CSV
    # has fewer rows, requests were lost (error file / expiry) and this chunk is incomplete.
    expected = batch.request_counts.total
    if len(results_df) != expected:
        print(f"  ⚠️  INCOMPLETE CHUNK: {len(results_df)} scored rows but {expected} "
              f"requests submitted ({expected - len(results_df)} missing/errored). "
              f"Do NOT treat this chunk as complete.")

    # Metadata-drift check: category/subcategory are re-derived from the CSV at retrieve time
    # and joined by metric_id.
    n_null_cat = int(results_df['category'].isna().sum())
    if n_null_cat:
        print(f"  ⚠️  {n_null_cat} rows have NULL category — the metrics CSV likely changed "
              f"since submit (metric_id join failed). Scores are valid; category labels are "
              f"not. Re-derive metadata from the CSV as it was AT SUBMIT time.")

    return results_df, output_path


def count_conversation_tokens(turns: list[dict], scale: Scale) -> int:
    """Count tokens for a conversation using OpenAI's token counting API."""
    import openai
    client = openai.OpenAI()

    conversation_text = format_turns_as_text(turns)
    messages = build_messages(
        conversation_text,
        "Example",
        "Example criterion description.",
        scale,
    )

    result = client.responses.input_tokens.count(
        model="gpt-4o-mini",
        input=messages,
    )
    return result.input_tokens


# BATCH TRANSPORT — Anthropic (Message Batches API - 50% cost reduction) Different transport
# from OpenAI: inline requests (no file upload), `system` is a top-level param, and custom_id
# must match ^[a-zA-Z0-9_-]{1,64}$ (no pipe, max 64) — so the OpenAI "{cid}|{metric_id}" id
# won't fit.

def create_anthropic_batch(tasks: list[tuple], scale: Scale, model: str, output_dir: Path,
                           offset: int = 0, limit: int | None = None) -> tuple[list[dict], Path]:
    """Build inline Anthropic batch requests AND write the id-map sidecar."""
    from datetime import datetime
    from collections import defaultdict

    model_name = model_name_for_filename(model)
    max_output_tokens = _max_completion_tokens_for(model_name)
    # One Structured-Outputs config for the whole batch (scale-wide, identical per request).
    output_config = build_score_output_config(scale)
    chunk_suffix = f"_o{offset}_l{limit}" if (offset or limit) else ""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    idmap_path = output_dir / f"batch_input_{model_name}{chunk_suffix}_{timestamp}_idmap.jsonl"

    # Group tasks by conversation_id (matches the OpenAI path; keeps a conv's requests
    # adjacent and formats the conversation text once per conversation).
    tasks_by_conv = defaultdict(list)
    for cid, turns, metric_row in tasks:
        tasks_by_conv[cid].append((turns, metric_row))

    requests = []
    idmap_records = []
    i = 0
    for cid, conv_tasks in tasks_by_conv.items():
        turns = conv_tasks[0][0]  # all tasks for same conv share turns
        conversation_text = format_turns_as_text(turns)
        for _, metric_row in conv_tasks:
            messages = build_messages(
                conversation_text,
                metric_row['Subcategory'],
                metric_row['Prompt'],
                scale,
            )
            # Split provider-neutral [system, user] into Anthropic's shape.
            system = messages[0]['content']
            user_messages = [messages[1]]

            custom_id = f"r{i}"
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model_name,
                    "max_tokens": max_output_tokens,
                    "system": system,
                    "messages": user_messages,
                    # Structured Outputs: force a JSON {"score": <int|"N/A">} conforming to
                    # the rubric enum, so no prose/stray-digit/out-of-range value can come
                    # back.
                    "output_config": output_config,
                },
            })
            idmap_records.append({
                "custom_id": custom_id,
                "conversation_id": cid,
                "metric_id": metric_row['metric_id'],
            })
            i += 1

    with open(idmap_path, 'w') as f:
        for rec in idmap_records:
            f.write(json.dumps(rec) + '\n')

    return requests, idmap_path


def submit_anthropic_batch(requests: list[dict]) -> str:
    """Create an Anthropic message batch from inline requests. Returns batch ID."""
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def check_anthropic_batch_status(batch_id: str) -> dict:
    """Check status of an Anthropic message batch."""
    import anthropic
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)

    counts = batch.request_counts
    # 'total' is the sum of terminal + in-flight; 'completed' counts only terminal-good
    # (succeeded), matching how the OpenAI dict is read downstream. errored/canceled/ expired
    # are surfaced together as 'failed'.
    total = (counts.processing + counts.succeeded + counts.errored
             + counts.canceled + counts.expired)
    failed = counts.errored + counts.canceled + counts.expired
    return {
        'id': batch.id,
        'status': batch.processing_status,
        'request_counts': {
            'total': total,
            'completed': counts.succeeded,
            'failed': failed,
        },
        'results_url': batch.results_url,
    }


def retrieve_anthropic_batch_results(batch_id: str, metrics: pd.DataFrame, output_dir: Path,
                                     scale: Scale, id_map: dict) -> tuple[pd.DataFrame, Path]:
    """Stream Anthropic batch results and convert to the SAME evaluation CSV + _raw.jsonl
    schema as the OpenAI path."""
    from datetime import datetime
    import anthropic
    client = anthropic.Anthropic()

    metric_lookup = {row['metric_id']: row for _, row in metrics.iterrows()}

    all_results = []
    raw_records = []
    seen_model = None

    # Stream results (don't load all in memory).
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        mapping = id_map.get(custom_id, {})
        conv_id = mapping.get('conversation_id')
        metric_id = mapping.get('metric_id')
        metric_row = metric_lookup.get(metric_id, {})

        result_type = result.result.type
        model_used = None
        error = None

        if result_type == 'succeeded':
            message = result.result.message
            model_used = message.model
            seen_model = seen_model or model_used
            # Structured Outputs returns the JSON payload ('{"score": 3}') in the text block.
            text = message.content[0].text if message.content else ''
            score, status = parse_score_json(text, scale)
            # If parsing found no score AND the reply was cut at max_tokens, it's a truncation
            # (Anthropic analog of OpenAI finish_reason == 'length'), not a parse error — a
            # cut-off payload is incomplete JSON, so parse_score_json returns 'error'; relabel
            # it as 'truncated' when the stop_reason confirms.
            if score is None and status != 'not_applicable' and message.stop_reason == 'max_tokens':
                status = 'truncated'
        else:
            # errored / canceled / expired — nothing scored; record the type for the raw log.
            score, status = None, 'error'
            error = result_type

        all_results.append({
            'conversation_id': conv_id,
            'metric_id': metric_id,
            'category': metric_row.get('Category'),
            'subcategory': metric_row.get('Subcategory'),
            'score': score,
            'score_status': status,
        })

        raw_records.append({
            'conversation_id': conv_id,
            'metric_id': metric_id,
            'source': metric_row.get('Source', ''),
            'category': metric_row.get('Category'),
            'subcategory': metric_row.get('Subcategory'),
            'model': model_used,
            'scale': scale.value if scale else None,
            'prompt': metric_row.get('Prompt'),
            'score': score,
            'score_status': status,
            'response_provider': 'anthropic-batch',
            'response_serialization': 'batch-passthrough',
            'response': result.model_dump(mode='json'),
            'error': error,
        })

    # Name the output from the model that actually scored the batch (fallback to the caller's
    # --model if no request succeeded), the batch id, and a timestamp.
    model_name = model_name_for_filename(seen_model) if seen_model else 'unknown'
    bid = batch_id.replace('msgbatch_', '').replace('batch_', '')[:12]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"efa_evaluation_results_{model_name}_batch_{bid}_{timestamp}.csv"
    raw_log_path = output_path.with_name(output_path.stem + '_raw.jsonl')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    with open(raw_log_path, 'w') as f:
        for rec in raw_records:
            f.write(json.dumps(rec, default=str) + '\n')
    print(f"  Raw LLM responses logged to {raw_log_path.name}")

    # Completeness check: the results stream should cover every request in the idmap.
    expected = len(id_map)
    if len(results_df) != expected:
        print(f"  ⚠️  INCOMPLETE CHUNK: {len(results_df)} result rows but {expected} "
              f"requests in the id-map ({expected - len(results_df)} missing). "
              f"Do NOT treat this chunk as complete.")

    # Non-succeeded requests (errored/canceled/expired) surfaced loudly, like the OpenAI
    # error-file count.
    n_failed = int((results_df['score_status'] == 'error').sum())
    if n_failed:
        print(f"  ⚠️  {n_failed} requests did not succeed (errored/canceled/expired) — "
              f"see score_status='error' rows and the raw log's 'error' field.")

    # Metadata-drift check (same rationale as the OpenAI path): NULL category means the
    # metrics CSV changed since submit so the metric_id join missed.
    n_null_cat = int(results_df['category'].isna().sum())
    if n_null_cat:
        print(f"  ⚠️  {n_null_cat} rows have NULL category — the metrics CSV likely changed "
              f"since submit (metric_id join failed). Scores are valid; category labels are "
              f"not. Re-derive metadata from the CSV as it was AT SUBMIT time.")

    return results_df, output_path


def load_id_map(id_map_path: Path) -> dict:
    """Load the idmap.jsonl sidecar into {custom_id: {conversation_id, metric_id}}."""
    id_map = {}
    with open(id_map_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            id_map[rec['custom_id']] = rec
    return id_map


# MODE DISPATCH (run_batch_mode / run_sync_mode)

def run_batch_mode(args):
    """Handle batch submit, status, or retrieve operations."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    provider = resolve_provider(args.model)
    if provider == 'local':
        print(f"Error: local models have no batch API. Omit --batch to run {args.model} "
              f"in sync mode (with --workers/--resume).")
        sys.exit(1)

    metrics_file = Path(args.metrics)
    # Output paths now route per artifact type via efa_result_dir(...) at each write site.

    if args.batch == 'submit':
        input_path = Path(args.input)
        scale = Scale.LIKERT if args.scale == 'likert' else Scale.BOOLEAN

        # Load conversations.
        strip_tool = not args.keep_tool_output
        chunk_suffix = f"_o{args.offset}_l{args.limit}" if (args.offset or args.limit) else ""
        metadata_path = efa_result_dir('convolearn_conversations') / f'convolearn_metadata{chunk_suffix}.csv'
        conversations = load_conversations(input_path, args.limit, strip_tool, metadata_path, offset=args.offset)

        print(f"\nLoading metrics from {metrics_file}...")
        metrics = load_metrics(metrics_file)
        print(f"Loaded {len(metrics)} metrics")

        # Always use the OpenAI counter for all providers: one shared yardstick keeps every
        # judge on the identical conversation set (needed for clean averaging).
        max_tokens = args.max_tokens if args.max_tokens > 0 else None
        if max_tokens:
            print(f"\nCounting tokens (max {max_tokens:,} per conversation)...")

            def count_tokens(item):
                cid, turns = item
                return cid, count_conversation_tokens(turns, scale)

            token_counts = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(count_tokens, (cid, turns)): cid
                          for cid, turns in conversations.items()}
                for i, future in enumerate(as_completed(futures)):
                    cid, tokens = future.result()
                    token_counts[cid] = tokens
                    if (i + 1) % 100 == 0:
                        print(f"  {i+1}/{len(conversations)}...")

            filtered = {cid: turns for cid, turns in conversations.items()
                       if token_counts[cid] <= max_tokens}
            skipped = len(conversations) - len(filtered)
            if skipped > 0:
                print(f"  Filtered out {skipped} conversations over {max_tokens:,} tokens")
            conversations = filtered
            print(f"  Using {len(conversations)} conversations")

        # Build task list
        tasks = []
        for cid, turns in conversations.items():
            for _, metric_row in metrics.iterrows():
                tasks.append((cid, turns, metric_row))

        total_tasks = len(tasks)
        print(f"\nCreating batch for {total_tasks} tasks ({len(conversations)} × {len(metrics)})...")

        batch_input_dir = efa_result_dir('batch_inputs', args.model)  # per-judge subfolder
        if provider == 'anthropic':
            requests, idmap_path = create_anthropic_batch(
                tasks, scale, args.model, batch_input_dir, offset=args.offset, limit=args.limit)
            print(f"Wrote id-map sidecar: {idmap_path}")
            print("\nSubmitting batch to Anthropic...")
            batch_id = submit_anthropic_batch(requests)
        else:
            batch_file = create_batch_file(tasks, scale, args.model, batch_input_dir,
                                           offset=args.offset, limit=args.limit,
                                           structured=args.structured)
            print(f"Created: {batch_file}")
            idmap_path = None
            print("\nSubmitting batch to OpenAI...")
            batch_id = submit_batch(batch_file)

        print("\n✓ Batch submitted successfully!")
        print(f"  Batch ID: {batch_id}")
        print("\nCheck status:")
        print(f"  python evaluate-metrics.py --batch status --batch-id {batch_id} --model {args.model}")
        print("\nRetrieve results (after completion):")
        retrieve_cmd = (f"  python evaluate-metrics.py --batch retrieve --batch-id {batch_id} "
                        f"--model {args.model} --metrics \"{metrics_file}\"")
        if idmap_path is not None:
            retrieve_cmd += f" --id-map \"{idmap_path}\""
        print(retrieve_cmd)

    elif args.batch == 'status':
        if not args.batch_id:
            print("Error: --batch-id required for status check")
            sys.exit(1)

        if provider == 'anthropic':
            status = check_anthropic_batch_status(args.batch_id)
        else:
            status = check_batch_status(args.batch_id)
        print(f"Batch ID: {status['id']}")
        print(f"Status: {status['status']}")
        counts = status['request_counts']
        print(f"Progress: {counts['completed']}/{counts['total']} completed, {counts['failed']} failed")
        # Each provider exposes only its own retrieval handle (OpenAI: file ids; Anthropic: results_url).
        if status.get('output_file_id'):
            print(f"Output file: {status['output_file_id']}")
        if status.get('error_file_id'):
            print(f"Error file: {status['error_file_id']}")
        if status.get('results_url'):
            print(f"Results URL: {status['results_url']}")

    elif args.batch == 'retrieve':
        if not args.batch_id:
            print("Error: --batch-id required for retrieval")
            sys.exit(1)

        print(f"Loading metrics from {metrics_file}...")
        metrics = load_metrics(metrics_file)

        print(f"Retrieving results for batch {args.batch_id}...")
        if args.scale == 'likert':
            scale = Scale.LIKERT
        elif args.scale == 'boolean':
            scale = Scale.BOOLEAN

        if provider == 'anthropic':
            if not args.id_map:
                print("Error: --id-map required to retrieve an Anthropic batch "
                      "(maps custom_id r{N} back to conversation_id/metric_id).")
                sys.exit(1)
            id_map = load_id_map(Path(args.id_map))
            results_df, output_path = retrieve_anthropic_batch_results(
                args.batch_id, metrics, efa_result_dir('scores'), scale, id_map)
        else:
            results_df, output_path = retrieve_batch_results(
                args.batch_id, metrics, efa_result_dir('scores'), scale=scale)

        print(f"\n✓ Saved {len(results_df)} results to {output_path}")
        print_status_summary(results_df, show_metrics=True)


def run_sync_mode(args):
    """Run synchronous evaluation with resumable batched writes."""
    from datetime import datetime

    input_path = Path(args.input)
    metrics_file = Path(args.metrics)
    scale = Scale.LIKERT if args.scale == 'likert' else Scale.BOOLEAN

    # Resume mode: append to existing CSV, skip already-completed conversations.
    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            print(f"Error: --resume file not found: {output_path}")
            sys.exit(1)

        existing = pd.read_csv(output_path)
        GOOD_STATUSES = {'scored', 'not_applicable'}
        # A conv is complete iff it has rows AND none of them are non-good.
        good_by_conv = existing.groupby('conversation_id')['score_status'].apply(
            lambda s: s.isin(GOOD_STATUSES).all()
        )
        completed_convs = set(good_by_conv[good_by_conv].index)
        failed_convs = set(good_by_conv[~good_by_conv].index)

        print(f"\nResuming from {output_path}")
        print(f"  {len(completed_convs)} conversations fully complete")
        if failed_convs:
            n_bad_rows = int((~existing['score_status'].isin(GOOD_STATUSES)).sum())
            print(f"  {len(failed_convs)} conversations have failures "
                  f"({n_bad_rows} error/truncated rows) — will re-run in full")
            # Purge ALL rows for failed convs from the CSV so the re-run doesn't duplicate.
            cleaned = existing[~existing['conversation_id'].isin(failed_convs)]
            cleaned.to_csv(output_path, index=False)
            print(f"  Purged {len(existing) - len(cleaned)} stale rows from {output_path.name}")

            # Purge the same convs from the raw JSONL sidecar to keep it consistent.
            raw_path = output_path.with_name(output_path.stem + '_raw.jsonl')
            if raw_path.exists():
                kept = []
                for line in raw_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # drop torn/truncated trailing line
                    if rec.get('conversation_id') not in failed_convs:
                        kept.append(line)
                raw_path.write_text('\n'.join(kept) + ('\n' if kept else ''))
                print(f"  Cleaned raw log {raw_path.name}")
    elif args.output is None:
        model_name = model_name_for_filename(args.model)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = efa_result_dir('scores') / f"efa_evaluation_results_{model_name}_{timestamp}.csv"
        completed_convs = set()
    else:
        output_path = Path(args.output)
        completed_convs = set()

    # Load conversations
    strip_tool = not args.keep_tool_output
    metadata_path = output_path.with_name(output_path.stem + '_metadata.csv')
    conversations = load_conversations(input_path, args.limit, strip_tool, metadata_path, offset=args.offset)

    # Skip already-completed conversations if resuming
    if completed_convs:
        before = len(conversations)
        conversations = {cid: turns for cid, turns in conversations.items() if cid not in completed_convs}
        print(f"  Skipping {before - len(conversations)} completed, {len(conversations)} remaining")

    print(f"\nLoading metrics from {metrics_file}...")
    metrics = load_metrics(metrics_file)
    print(f"Loaded {len(metrics)} metrics")

    # Filter by token count if max_tokens is set.
    provider = resolve_provider(args.model)
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    if max_tokens and provider == 'local':
        print("\nSkipping token prefilter for local judge (no OpenAI counter).")
        max_tokens = None
    token_filter_stats = None  # populated below; used in end-of-run summary
    if max_tokens:
        print(f"\nCounting tokens (max {max_tokens:,} per conversation)...")

        def count_tokens(item):
            cid, turns = item
            return cid, count_conversation_tokens(turns, scale)

        token_counts = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(count_tokens, (cid, turns)): cid
                      for cid, turns in conversations.items()}
            for i, future in enumerate(as_completed(futures)):
                cid, tokens = future.result()
                token_counts[cid] = tokens
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(conversations)}...")

        # Filter conversations
        filtered = {cid: turns for cid, turns in conversations.items()
                   if token_counts[cid] <= max_tokens}
        n_skipped = len(conversations) - len(filtered)
        if n_skipped > 0:
            print(f"  Filtered out {n_skipped} conversations over {max_tokens:,} tokens")
        conversations = filtered
        print(f"  Using {len(conversations)} conversations")

        token_filter_stats = {'n_skipped': n_skipped}

    total_tasks = len(conversations) * len(metrics)
    print(f"\nEvaluating {total_tasks} tasks ({len(conversations)} conversations × {len(metrics)} metrics)")
    print(f"Parallelizing by conversation (max {args.workers} workers) to maximize prompt caching")

    start_time = time.monotonic()

    # Raw response log: one JSONL record per LLM call, written immediately on return.
    raw_log_path = output_path.with_name(output_path.stem + '_raw.jsonl')
    raw_log_lock = threading.Lock()

    def evaluate_conversation(conv_id: str, turns: list[dict]) -> list[dict]:
        """Evaluate all metrics for a single conversation sequentially (for caching)."""
        results = []
        for _, metric_row in metrics.iterrows():
            result = evaluate_single_metric(
                conv_id, turns,
                metric_row['metric_id'],
                metric_row['Category'],
                metric_row['Subcategory'],
                metric_row['Prompt'],
                args.model, scale,
                source=metric_row.get('Source', ''),
                raw_log_path=raw_log_path,
                raw_log_lock=raw_log_lock,
                base_url=args.base_url,
                structured=args.structured,
            )
            results.append(result)
        return results

    batch_buffer = []
    completed_convs = 0
    batch_size = args.workers
    # First write includes header only if the file doesn't exist yet (fresh run, not resume)
    write_header = not output_path.exists()

    def flush_batch():
        """Append buffered results to CSV and clear buffer."""
        nonlocal write_header
        if not batch_buffer:
            return
        batch_df = pd.DataFrame(batch_buffer)
        batch_df.to_csv(output_path, mode='a', header=write_header, index=False)
        write_header = False
        batch_buffer.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_conversation, cid, turns): cid
            for cid, turns in conversations.items()
        }

        for future in as_completed(futures):
            cid = futures[future]
            try:
                conv_results = future.result()
                batch_buffer.extend(conv_results)
            except Exception as e:
                print(f"  ❌ {cid}: {e}")
                for _, metric_row in metrics.iterrows():
                    batch_buffer.append({
                        'conversation_id': cid,
                        'metric_id': metric_row['metric_id'],
                        'source': metric_row.get('Source', ''),
                        'category': metric_row['Category'],
                        'subcategory': metric_row['Subcategory'],
                        'score': None,
                        'score_status': 'error'
                    })

            completed_convs += 1
            completed_tasks = completed_convs * len(metrics)
            print(f"  Progress: {completed_convs}/{len(conversations)} conversations ({completed_tasks}/{total_tasks} tasks)")

            # Flush after each batch of `workers` conversations
            if completed_convs % batch_size == 0:
                flush_batch()
                print(f"  ↓ Checkpointed to {output_path.name}")

    # Flush any remaining conversations
    flush_batch()

    elapsed = time.monotonic() - start_time

    # Load full CSV for final summary (includes resumed rows)
    results_df = pd.read_csv(output_path)
    print(f"\n✓ Saved {len(results_df)} results to {output_path}")
    if raw_log_path.exists():
        print(f"  Raw LLM responses logged to {raw_log_path.name}")
    print_status_summary(results_df, show_metrics=True)
    if token_filter_stats:
        print(f"  Skipped {token_filter_stats['n_skipped']} conversations exceeding max token count ({args.max_tokens:,} tokens)")
    print(f"  Elapsed: {elapsed/60:.1f} min ({elapsed:.0f}s)")


# CLI

def main():
    parser = argparse.ArgumentParser(description='Evaluate conversations for EFA')

    # Common arguments
    parser.add_argument('--metrics', type=str, default=None,
                        help='Metrics CSV path (default: Tutor Metrics.csv next to this script)')
    parser.add_argument('--model', type=str, default='openai:/gpt-5-nano', help='Judge model')
    parser.add_argument('--scale', choices=['likert', 'boolean'], default='likert', help='Evaluation scale')
    parser.add_argument('--limit', type=int, default=None, help='Max conversations to evaluate (after --offset)')
    parser.add_argument('--offset', type=int, default=0, help='Skip the first N conversations. For chunked batch runs that must stay under OpenAI batch caps, e.g. --offset 250 --limit 250 = conversations 250-499')
    parser.add_argument('--max-tokens', type=int, default=25000, help='Max tokens per conversation (default: 25000, 0=no limit)')
    parser.add_argument('--keep-tool-output', action='store_true', help='Keep tool call artifacts in tutor messages (default: strip them)')

    # Synchronous mode arguments
    parser.add_argument('--input', type=str, help='Conversation source: directory with JSON files, or CSV file (e.g., ConvoLearn)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--workers', type=int, default=10, help='Parallel workers (sync mode). Also the batch size for incremental CSV writes.')
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing results CSV: skips conversations already present in the file')
    parser.add_argument('--base-url', type=str, default=None, help='Local OpenAI-compatible server URL for a local:/ model (e.g. http://localhost:11434/v1); sync mode only')
    parser.add_argument('--structured', action='store_true', help='Force Structured Outputs (JSON-schema-constrained {"score": ...}) on the openai:/ and local:/ paths. Off by default (free-text integer/N-A). The anthropic:/ path always uses Structured Outputs regardless of this flag.')

    # Batch mode arguments — ~50% cheaper than sync (no prompt caching to forfeit on short
    # conversations), at the cost of async turnaround (up to 24h) and no live
    # checkpoint/resume. submit -> status -> retrieve.
    parser.add_argument('--batch', choices=['submit', 'status', 'retrieve'],
                        help='Batch mode: submit job, check status, or retrieve results')
    parser.add_argument('--batch-id', type=str, help='Batch ID (for status/retrieve)')
    parser.add_argument('--id-map', type=str, default=None, help='Anthropic retrieve: path to the idmap.jsonl sidecar written at submit (maps custom_id r{N} to conversation_id/metric_id)')

    args = parser.parse_args()

    # Default to the metrics CSV next to this script.
    if args.metrics is None:
        args.metrics = str(Path(__file__).parent / 'Tutor Metrics.csv')

    if args.batch:
        if args.batch == 'submit' and not args.input:
            parser.error("--input is required for --batch submit")
        run_batch_mode(args)
    else:
        if not args.input:
            parser.error("--input is required")
        run_sync_mode(args)


if __name__ == "__main__":
    main()
