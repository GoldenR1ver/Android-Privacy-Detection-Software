import argparse
import ast
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request


SYSTEM_PROMPT = (
    "You are an expert in analyzing the correctness, completeness, and consistency "
    "between Android Data Safety declaration and Privacy Policy."
)


USER_PROMPT_TEMPLATE = """Let's compare and analyze the information between Data Safety and Privacy Policy.

Classify three labels:
1) incorrect: Data Safety does NOT provide information, but Privacy Policy mentions it.
2) incomplete: Data Safety provides information, but less complete than Privacy Policy.
3) inconsistent: Data Safety is provided, but conflicts with Privacy Policy.

Few-shot examples for conflict judgment of each label:

[incorrect example -> 1]
Data Safety: "No data shared. No data collected."
Privacy Policy: "We collect device identifiers and share them with analytics providers."
Reason: Data Safety omits a data practice explicitly stated in Privacy Policy.
Output: {{"incorrect": 1, "incomplete": 0, "inconsistent": 0}}

[incorrect example -> 0]
Data Safety: "Location is collected."
Privacy Policy: "We collect approximate location."
Reason: Both mention collection; this is not a missing-disclosure conflict.
Output: {{"incorrect": 0, "incomplete": 0, "inconsistent": 0}}

[incomplete example -> 1]
Data Safety: "Personal info is collected for app functionality."
Privacy Policy: "We collect name, email, phone, and address for account, support, and fraud prevention."
Reason: Data Safety has the same direction but clearly less complete coverage.
Output: {{"incorrect": 0, "incomplete": 1, "inconsistent": 0}}

[incomplete example -> 0]
Data Safety: "Contacts are collected for social features."
Privacy Policy: "Contacts are collected for social features."
Reason: Information granularity is aligned.
Output: {{"incorrect": 0, "incomplete": 0, "inconsistent": 0}}

[inconsistent example -> 1]
Data Safety: "No location data is collected."
Privacy Policy: "We collect precise GPS location."
Reason: Direct contradiction between two sources.
Output: {{"incorrect": 0, "incomplete": 0, "inconsistent": 1}}

[inconsistent example -> 0]
Data Safety: "Device ID is collected and shared with advertisers."
Privacy Policy: "We collect device ID and share with ad partners."
Reason: Statements are mutually consistent.
Output: {{"incorrect": 0, "incomplete": 0, "inconsistent": 0}}

Output format strictly as JSON:
{{"incorrect": 0 or 1, "incomplete": 0 or 1, "inconsistent": 0 or 1}}
Return JSON only.

Data Safety:
{data_safety}

Privacy Policy:
{privacy_policy}
"""


def build_user_prompt(data_safety: str, privacy_policy: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        data_safety=data_safety,
        privacy_policy=privacy_policy,
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def count_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


class RuntimeLogger:
    def __init__(self, log_file: Optional[Path]) -> None:
        self._log_fp = None
        if log_file is not None:
            ensure_parent_dir(log_file)
            self._log_fp = log_file.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        if self._log_fp is not None:
            self._log_fp.write(line + "\n")
            self._log_fp.flush()

    def close(self) -> None:
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None


def normalize_prediction(d: Dict) -> Dict[str, int]:
    return {
        "incorrect": int(d.get("incorrect", 0)),
        "incomplete": int(d.get("incomplete", 0)),
        "inconsistent": int(d.get("inconsistent", 0)),
    }


def extract_json_dict(text: str) -> Dict[str, int]:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    for candidate in [stripped]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return normalize_prediction(parsed)
        except Exception:
            pass

    match = re.search(r"\{[\s\S]*?\}", stripped)
    if match:
        fragment = match.group(0)
        try:
            parsed = json.loads(fragment)
            if isinstance(parsed, dict):
                return normalize_prediction(parsed)
        except Exception:
            try:
                parsed = ast.literal_eval(fragment)
                if isinstance(parsed, dict):
                    return normalize_prediction(parsed)
            except Exception:
                pass

    return {"incorrect": 0, "incomplete": 0, "inconsistent": 0}


class BaseProvider:
    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, int]:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, int]:
        ds = data_safety.lower()
        pp = privacy_policy.lower()
        has_no_data = ("no data" in pp) or ("content not provided" in pp)
        ds_empty = ("'data_shared': []" in ds and "'data_collected': []" in ds)

        incorrect = 1 if ds_empty and not has_no_data else 0
        incomplete = 1 if ds_empty and not has_no_data else 0
        inconsistent = 0
        return {"incorrect": incorrect, "incomplete": incomplete, "inconsistent": inconsistent}


class DeepSeekProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-reasoner",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("DeepSeek API key is empty.")
        try:
            self.api_key.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "DeepSeek API key contains non-ASCII characters. "
                "Please set a real API key (do not use placeholders like Chinese text)."
            ) from exc
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, int]:
        user_prompt = build_user_prompt(data_safety, privacy_policy)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "Failed to build HTTP headers. Check DEEPSEEK_API_KEY for invalid characters."
            ) from exc

        obj = json.loads(raw)
        content = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return extract_json_dict(content)


class LocalHFProvider(BaseProvider):
    def __init__(
        self,
        model_id: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 256,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

    def infer(self, data_safety: str, privacy_policy: str) -> Dict[str, int]:
        user_prompt = build_user_prompt(data_safety, privacy_policy)

        try:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        out = self.generator(prompt)
        generated = out[0]["generated_text"] if out else "{}"
        return extract_json_dict(generated)


def make_provider(args: argparse.Namespace) -> BaseProvider:
    if args.provider == "mock":
        return MockProvider()
    if args.provider == "local":
        return LocalHFProvider(
            model_id=args.local_model_id,
            max_new_tokens=args.max_new_tokens,
        )
    if args.provider == "deepseek":
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set and --api-key was not provided.")
        return DeepSeekProvider(
            api_key=api_key,
            model=args.deepseek_model,
            base_url=args.base_url,
            timeout=args.timeout,
        )
    raise ValueError(f"Unknown provider: {args.provider}")


def run_audit(
    input_csv: Path,
    output_csv: Path,
    provider: BaseProvider,
    limit: Optional[int],
    logger: RuntimeLogger,
    log_every: int,
    resume: bool = False,
    resume_from: Optional[int] = None,
) -> None:
    logger.log(
        f"[audit] start: input={input_csv}, output={output_csv}, limit={limit}, "
        f"resume={resume}, resume_from={resume_from}"
    )
    ensure_parent_dir(output_csv)
    if resume_from is not None and resume_from < 1:
        raise ValueError("--resume-from must be >= 1.")

    start_row = 1
    if resume_from is not None:
        start_row = resume_from
        logger.log(f"[audit] manual resume: start from row {start_row}")
    elif resume:
        completed = count_data_rows(output_csv)
        start_row = completed + 1
        logger.log(f"[audit] auto resume: detected {completed} completed rows, start from row {start_row}")

    write_mode = "a" if (start_row > 1 and output_csv.exists()) else "w"
    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        write_mode, newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if "result" not in fieldnames:
            fieldnames.append("result")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()

        processed_count = 0
        total_seen = 0
        for row in reader:
            total_seen += 1
            if total_seen < start_row:
                continue
            if limit is not None and total_seen > limit:
                logger.log(f"[audit] reached limit={limit}, stop")
                break
            data_safety = row.get("data_safety_content", "")
            privacy_policy = row.get("privacy_policy_content", "")
            logger.log(f"[audit] row {total_seen}: request model inference")
            pred = provider.infer(data_safety, privacy_policy)
            logger.log(f"[audit] row {total_seen}: inference finished")
            row["result"] = json.dumps(pred, ensure_ascii=False)
            writer.writerow(row)
            processed_count += 1
            if processed_count % log_every == 0:
                logger.log(f"[audit] processed row {total_seen}")

    if total_seen < start_row:
        logger.log(
            f"[audit] warning: start_row={start_row} exceeds input rows={total_seen}, "
            "nothing processed"
        )
    logger.log(
        f"[audit] completed: processed_in_this_run={processed_count}, "
        f"last_input_row_seen={total_seen}"
    )


def postprocess_results(input_csv: Path, output_csv: Path, logger: RuntimeLogger) -> None:
    logger.log(f"[postprocess] start: input={input_csv}, output={output_csv}")
    ensure_parent_dir(output_csv)
    with input_csv.open("r", newline="", encoding="utf-8") as fin, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if "result" in fieldnames:
            fieldnames.remove("result")
        for c in ["incorrect", "incomplete", "inconsistent"]:
            if c not in fieldnames:
                fieldnames.append(c)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            pred = extract_json_dict(row.get("result", ""))
            row.pop("result", None)
            row.update({k: str(v) for k, v in pred.items()})
            writer.writerow(row)
            logger.log(f"[postprocess] processed row {idx}")
    logger.log("[postprocess] completed")


def evaluate_results(
    prediction_csv: Path,
    groundtruth_csv: Path,
    metrics_output_csv: Path,
    logger: RuntimeLogger,
    figures_dir: Optional[Path] = None,
) -> None:
    logger.log(
        f"[evaluate] start: prediction={prediction_csv}, groundtruth={groundtruth_csv}, "
        f"metrics={metrics_output_csv}, figures_dir={figures_dir}"
    )
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    plt = None
    if figures_dir is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt

    labels = ["incorrect", "incomplete", "inconsistent"]
    pred_df = pd.read_csv(prediction_csv)
    gt_df = pd.read_csv(groundtruth_csv)

    for c in labels:
        pred_df[c] = pred_df[c].astype(int)
        gt_df[c] = gt_df[c].astype(int)

    rows = []
    for c in labels:
        y_true = gt_df[c]
        y_pred = pred_df[c]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        rows.append(
            {
                "label": c,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1]),
            }
        )
        logger.log(
            f"[evaluate] {c}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, accuracy={accuracy:.4f}"
        )

        if figures_dir is not None and plt is not None:
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(4.5, 4.0))
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix - {c}")
            plt.colorbar()
            plt.xticks([0, 1], ["0", "1"])
            plt.yticks([0, 1], ["0", "1"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            fig_path = figures_dir / f"cm_{c}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()

    ensure_parent_dir(metrics_output_csv)
    with metrics_output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.log(f"[evaluate] metrics written to: {metrics_output_csv}")
    logger.log("[evaluate] completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Android privacy audit pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    common_runtime = argparse.ArgumentParser(add_help=False)
    common_runtime.add_argument(
        "--log-file",
        default="results/run_audit_runtime.log",
        help="Path to runtime log file",
    )

    common_llm = argparse.ArgumentParser(add_help=False)
    common_llm.add_argument(
        "--provider",
        choices=["local", "deepseek", "mock"],
        default="mock",
        help="LLM backend provider",
    )
    common_llm.add_argument("--local-model-id", default="microsoft/Phi-3-mini-4k-instruct")
    common_llm.add_argument("--max-new-tokens", type=int, default=256)
    common_llm.add_argument("--api-key", default=None, help="DeepSeek API key")
    common_llm.add_argument("--deepseek-model", default="deepseek-reasoner")
    common_llm.add_argument("--base-url", default="https://api.deepseek.com/chat/completions")
    common_llm.add_argument("--timeout", type=int, default=120)

    p_audit = sub.add_parser(
        "audit",
        parents=[common_llm, common_runtime],
        help="Run raw model audit",
    )
    p_audit.add_argument("--input-csv", required=True)
    p_audit.add_argument("--output-csv", required=True)
    p_audit.add_argument("--limit", type=int, default=None)
    p_audit.add_argument("--log-every", type=int, default=1)
    p_audit.add_argument("--resume", action="store_true", help="Resume from existing output CSV")
    p_audit.add_argument("--resume-from", type=int, default=None, help="1-based input row index to resume from")

    p_post = sub.add_parser("postprocess", parents=[common_runtime], help="Parse result column to labels")
    p_post.add_argument("--input-csv", required=True)
    p_post.add_argument("--output-csv", required=True)

    p_eval = sub.add_parser("evaluate", parents=[common_runtime], help="Evaluate predictions against groundtruth")
    p_eval.add_argument("--prediction-csv", required=True)
    p_eval.add_argument("--groundtruth-csv", required=True)
    p_eval.add_argument("--metrics-output-csv", required=True)
    p_eval.add_argument("--figures-dir", default=None)

    p_full = sub.add_parser("full", parents=[common_llm, common_runtime], help="Run full pipeline")
    p_full.add_argument("--input-csv", required=True)
    p_full.add_argument("--groundtruth-csv", required=True)
    p_full.add_argument("--raw-output-csv", required=True)
    p_full.add_argument("--processed-output-csv", required=True)
    p_full.add_argument("--metrics-output-csv", required=True)
    p_full.add_argument("--figures-dir", default=None)
    p_full.add_argument("--limit", type=int, default=None)
    p_full.add_argument("--log-every", type=int, default=1)
    p_full.add_argument("--resume", action="store_true", help="Resume from existing raw output CSV")
    p_full.add_argument("--resume-from", type=int, default=None, help="1-based input row index to resume from")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logger = RuntimeLogger(Path(args.log_file))

    try:
        logger.log(f"[main] command={args.command}")

        if args.command == "audit":
            provider = make_provider(args)
            run_audit(
                Path(args.input_csv),
                Path(args.output_csv),
                provider,
                args.limit,
                logger,
                max(1, int(args.log_every)),
                args.resume,
                args.resume_from,
            )
            return 0

        if args.command == "postprocess":
            postprocess_results(Path(args.input_csv), Path(args.output_csv), logger)
            return 0

        if args.command == "evaluate":
            evaluate_results(
                Path(args.prediction_csv),
                Path(args.groundtruth_csv),
                Path(args.metrics_output_csv),
                logger,
                Path(args.figures_dir) if args.figures_dir else None,
            )
            return 0

        if args.command == "full":
            provider = make_provider(args)
            run_audit(
                Path(args.input_csv),
                Path(args.raw_output_csv),
                provider,
                args.limit,
                logger,
                max(1, int(args.log_every)),
                args.resume,
                args.resume_from,
            )
            postprocess_results(Path(args.raw_output_csv), Path(args.processed_output_csv), logger)
            evaluate_results(
                Path(args.processed_output_csv),
                Path(args.groundtruth_csv),
                Path(args.metrics_output_csv),
                logger,
                Path(args.figures_dir) if args.figures_dir else None,
            )
            return 0
    finally:
        logger.log("[main] finished")
        logger.close()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
