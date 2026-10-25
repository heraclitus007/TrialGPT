__author__ = "trial-jev"

"""
Criterion-level eligibility with Jev (TypeSafe System One), benchmarked on the
TrialGPT expert-annotated parquet.

The parquet is only read. Predictions go to trial-jev/jev_eligibility.csv: every
parquet column, expert_evidence (the note sentences behind expert_sentences, as
text), plus, per run (the prefix defaults to "jev"):
	{prefix}_eligibility	predicted label
	{prefix}_confidence	Jev confidence
	{prefix}_p_<label>	Jev probability of each option, one column per label:
				included / not included			(inclusion rows only)
				excluded / not excluded			(exclusion rows only)
				not enough information / not applicable	(both)
Re-running skips rows that already have a prediction in the CSV; if every row has
one, it makes no Jev calls and just refreshes the CSV.

usage:
	python trial-jev/jev_eligibility.py
	python trial-jev/jev_eligibility.py --limit 3 --dry_run
	python trial-jev/jev_eligibility.py --output trial-jev/jev_eligibility_v2.csv
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tqdm

PARQUET = Path(__file__).resolve().parent.parent / "train-00000-of-00001.parquet"
OUTPUT = Path(__file__).resolve().parent / "jev_eligibility.csv"

# label -> definition, taken from the TrialGPT-Matching prompt (trialgpt_matching/TrialGPT.py)
LABELS = {
	"inclusion": {
		"included": "The patient meets this inclusion criterion.",
		"not included": "The patient does not meet this inclusion criterion.",
		"not enough information": "The patient note does not contain sufficient information to classify this criterion.",
		"not applicable": "The criterion is not applicable to the patient: the patient does not meet the premise of the criterion. This is not very common.",
	},
	"exclusion": {
		"excluded": "The patient meets this exclusion criterion and should be excluded from the trial.",
		"not excluded": "The patient does not meet this exclusion criterion.",
		"not enough information": "The patient note does not contain sufficient information to classify this criterion.",
		"not applicable": "The criterion is not applicable to the patient: the patient does not meet the premise of the criterion. This is not very common.",
	},
}

ALL_LABELS = ["included", "not included", "excluded", "not excluded", "not enough information", "not applicable"]

HINT = "Use \"not enough information\" as little as possible: if the note does not mention a medically important fact, assume that the fact is not true for the patient."

def cited_sentences(note: str, sentence_ids: str) -> str:
	"""The note sentences behind a list of sentence ids, e.g. "[0, 8]" -> "0. A 58-year-old ...\n8. The patient will ..."."""
	sentences = note.split("\n")  # the note is numbered one sentence per line: "0. ...", "1. ..."
	return "\n".join(sentences[i] for i in json.loads(sentence_ids))


def load(output: Path = OUTPUT) -> pd.DataFrame:
	"""The parquet (plus the expert evidence sentences as text), and any prediction columns already saved in the output CSV."""
	df = pd.read_parquet(PARQUET)
	df.insert(
		df.columns.get_loc("expert_sentences") + 1,
		"expert_evidence",
		[cited_sentences(note, ids) for note, ids in zip(df["note"], df["expert_sentences"])],
	)
	if output.exists():
		saved = pd.read_csv(output)
		extra = [c for c in saved.columns if c not in df.columns]
		df = df.merge(saved[["annotation_id"] + extra], on="annotation_id", how="left")
	return df


def question_text(criterion_type: str, criterion_text: str) -> str:
	return (
		f"This is one {criterion_type} criterion of a clinical trial: \"{criterion_text}\". "
		f"Classify the patient's eligibility for this specific {criterion_type} criterion based on the patient note. "
		+ HINT
	)


def build_pairs(df: pd.DataFrame) -> list:
	"""One unit of work per (patient, trial): shared state + one question per criterion row."""
	pairs = []
	for (patient_id, trial_id), group in df.groupby(["patient_id", "trial_id"], sort=False):
		first = group.iloc[0]
		state = {
			"patient_note": first["note"],
			"trial_title": first["trial_title"],
		}
		questions = {}
		for _, row in group.iterrows():
			questions[f"c{row['annotation_id']}"] = {
				"annotation_id": int(row["annotation_id"]),
				"criterion_type": row["criterion_type"],
				"instructions": question_text(row["criterion_type"], row["criterion_text"]),
				"labels": LABELS[row["criterion_type"]],
			}
		pairs.append({"patient_id": patient_id, "trial_id": trial_id, "state": state, "questions": questions})
	return pairs


def make_jev_runner():
	from typesafe_sdk import Choice, TypeSafeClient

	client = TypeSafeClient()
	input_tokens = []

	def run(pair: dict) -> dict:
		questions = {
			key: Choice(instructions=q["instructions"], criteria=q["labels"])
			for key, q in pair["questions"].items()
		}
		response = client.system_one(state=pair["state"], questions=questions)
		input_tokens.append(response.usage.input_tokens)

		out = {}
		for key in pair["questions"]:
			answer = response.choices[key]
			out[key] = {
				"label": answer.choice,
				"confidence": answer.confidence,
				"probabilities": answer.probabilities,
			}
		return out

	return run, client, input_tokens


def call_with_retry(run, pair, retries=4):
	for attempt in range(retries):
		try:
			return run(pair)
		except Exception as e:
			if attempt == retries - 1:
				raise
			wait = 2 ** attempt * 5
			print(f"\n{pair['patient_id']} {pair['trial_id']}: {type(e).__name__}: {e} -- retrying in {wait}s")
			time.sleep(wait)


def prob_column(prefix: str, label: str) -> str:
	return f"{prefix}_p_{label.replace(' ', '_')}"


def write_columns(predictions: dict, prefix: str, output: Path = OUTPUT):
	"""Merge predictions (annotation_id -> answer) into the output CSV, touching only the prefix columns."""
	df = load(output)
	label_col = f"{prefix}_eligibility"
	conf_col = f"{prefix}_confidence"
	prob_cols = {label: prob_column(prefix, label) for label in ALL_LABELS}

	if label_col not in df.columns:
		df[label_col] = pd.Series([None] * len(df), dtype="object")
	for col in [conf_col] + list(prob_cols.values()):
		if col not in df.columns:
			df[col] = np.nan

	idx = df.index[df["annotation_id"].isin(predictions)]
	for i in idx:
		answer = predictions[int(df.at[i, "annotation_id"])]
		df.at[i, label_col] = answer["label"]
		df.at[i, conf_col] = answer["confidence"]
		# only the 4 options of this criterion's type get a value; the other 2 stay blank
		for label, p in answer["probabilities"].items():
			df.at[i, prob_cols[label]] = p

	tmp = output.with_suffix(".csv.tmp")
	df.to_csv(tmp, index=False)
	os.replace(tmp, output)
	return [label_col, conf_col] + list(prob_cols.values()), len(idx)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--prefix", default="jev", help="column prefix (default: jev)")
	parser.add_argument("--output", type=Path, default=OUTPUT, help=f"predictions CSV (default: {OUTPUT.name})")
	parser.add_argument("--limit", type=int, help="only run the first N patient-trial pairs")
	parser.add_argument("--workers", type=int, default=8)
	parser.add_argument("--overwrite", action="store_true", help="re-run rows that already have a prediction")
	parser.add_argument("--dry_run", action="store_true", help="print the inputs for the first pair and exit")
	args = parser.parse_args()

	prefix = args.prefix
	df = load(args.output)

	col = f"{prefix}_eligibility"
	if col in df.columns and not args.overwrite:
		df = df[df[col].isna()]
	pairs = build_pairs(df)
	if args.limit:
		pairs = pairs[:args.limit]

	if args.dry_run:
		pair = pairs[0]
		print(f"{len(pairs)} pairs, {sum(len(p['questions']) for p in pairs)} criteria to run\n")
		print(json.dumps({"state": pair["state"], "questions": {k: {"instructions": q["instructions"], "criteria": q["labels"]} for k, q in pair["questions"].items()}}, indent=2))
		raise SystemExit

	if not pairs:
		# no Jev calls needed; still rewrite the CSV so derived columns (e.g. expert_evidence) are current
		write_columns({}, prefix, args.output)
		print(f"nothing to run: every row already has {col} (use --overwrite to re-run); refreshed {args.output}")
		raise SystemExit

	run, client, input_tokens = make_jev_runner()

	predictions = {}
	failed = []
	start = time.time()
	with ThreadPoolExecutor(max_workers=args.workers) as pool:
		futures = {pool.submit(call_with_retry, run, pair): pair for pair in pairs}
		for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
			pair = futures[future]
			try:
				answers = future.result()
			except Exception as e:
				failed.append((pair["patient_id"], pair["trial_id"], f"{type(e).__name__}: {e}"))
				continue
			for key, answer in answers.items():
				predictions[pair["questions"][key]["annotation_id"]] = answer

	client.close()

	cols, n = write_columns(predictions, prefix, args.output)
	print(f"\njev: wrote {n} rows to {args.output} -> {cols} in {time.time() - start:.1f}s")
	print(f"{len(input_tokens)} calls, {sum(input_tokens):,} input tokens")
	for patient_id, trial_id, error in failed:
		print(f"FAILED {patient_id} {trial_id}: {error}")
