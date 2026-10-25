__author__ = "trial-jev"

"""
Score criterion-level eligibility predictions (trial-jev/jev_eligibility.csv) against the expert
(physician consensus) labels, and compare the models with each other.

Every `{prefix}_eligibility` column is treated as a model:
	gpt4	GPT-4 predictions shipped with the dataset (the TrialGPT paper numbers)
	jev	written by jev_eligibility.py

usage:
	python trial-jev/jev_score.py
	python trial-jev/jev_score.py --split heldout
	python trial-jev/jev_score.py --models jev gpt4 --confusion
	python trial-jev/jev_score.py --input trial-jev/jev_eligibility_v2.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

from jev_eligibility import ALL_LABELS, LABELS, OUTPUT, load, prob_column

TRUTH = "expert_eligibility"
TYPES = ["inclusion", "exclusion"]


def model_columns(df: pd.DataFrame) -> list:
	return [c[: -len("_eligibility")] for c in df.columns if c.endswith("_eligibility") and c != TRUTH]


def split_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
	if split == "train":
		return df[df["training"]]
	if split == "heldout":
		return df[~df["training"]]
	return df


def summary(df: pd.DataFrame, model: str) -> dict:
	pred = df[f"{model}_eligibility"]
	answered = pred.notna()
	correct = pred == df[TRUTH]

	row = {
		"model": model,
		"coverage": f"{answered.sum()}/{len(df)}",
		# unanswered rows count as wrong, so partial runs cannot look better than they are
		"acc": correct.mean(),
	}
	for t in TYPES:
		mask = df["criterion_type"] == t
		row[f"acc_{t[:3]}"] = correct[mask].mean()
		row[f"macroF1_{t[:3]}"] = f1_score(
			df.loc[mask, TRUTH], pred[mask].fillna("<none>"),
			labels=list(LABELS[t]), average="macro", zero_division=0,
		)
	return row


def per_class(df: pd.DataFrame, model: str) -> pd.DataFrame:
	rows = []
	pred = df[f"{model}_eligibility"].fillna("<none>")
	for t in TYPES:
		mask = df["criterion_type"] == t
		labels = list(LABELS[t])
		p, r, f, n = precision_recall_fscore_support(
			df.loc[mask, TRUTH], pred[mask], labels=labels, zero_division=0,
		)
		for label, pi, ri, fi, ni in zip(labels, p, r, f, n):
			rows.append({"type": t, "label": label, "n": ni, "precision": pi, "recall": ri, "f1": fi})
	return pd.DataFrame(rows)


def confusion(df: pd.DataFrame, model: str) -> None:
	pred = df[f"{model}_eligibility"].fillna("<none>")
	for t in TYPES:
		mask = df["criterion_type"] == t
		labels = list(LABELS[t])
		cm = confusion_matrix(df.loc[mask, TRUTH], pred[mask], labels=labels)
		print(f"\n{model} / {t} (rows = expert, cols = predicted)")
		print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


def calibration(df: pd.DataFrame, model: str, bins: int = 5) -> None:
	conf = df.get(f"{model}_confidence")
	if conf is None or conf.notna().sum() == 0:
		return
	mask = conf.notna()
	conf = conf[mask]
	correct = (df.loc[mask, f"{model}_eligibility"] == df.loc[mask, TRUTH]).astype(float)

	edges = np.linspace(0, 1, bins + 1)
	bucket = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
	rows = []
	ece = 0.0
	for b in range(bins):
		sel = bucket == b
		if sel.sum() == 0:
			continue
		ece += sel.sum() / len(conf) * abs(correct[sel].mean() - conf[sel].mean())
		rows.append({"confidence": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(sel.sum()),
			"mean_conf": conf[sel].mean(), "accuracy": correct[sel].mean()})
	print(f"\n{model} calibration (ECE = {ece:.4f})")
	print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))


def option_probabilities(df: pd.DataFrame, model: str) -> None:
	"""Mean probability the model gave each option, grouped by the expert label."""
	for t in TYPES:
		labels = list(LABELS[t])
		cols = [prob_column(model, label) for label in labels]
		if not all(c in df for c in cols):
			return
		sub = df[(df["criterion_type"] == t) & df[cols].notna().any(axis=1)]
		if sub.empty:
			continue
		table = sub.groupby(TRUTH)[cols].mean().reindex(labels)
		table.columns = labels
		table.insert(0, "n", sub[TRUTH].value_counts().reindex(labels).fillna(0).astype(int))
		print(f"\n{model} / {t}: mean probability per option (rows = expert label)")
		print(table.to_string(float_format="%.4f"))


def agreement(df: pd.DataFrame, models: list) -> pd.DataFrame:
	"""Pairwise % agreement (upper) and Cohen's kappa (lower) between models and the experts."""
	names = [TRUTH] + [f"{m}_eligibility" for m in models]
	short = ["expert"] + models
	out = pd.DataFrame(index=short, columns=short, dtype=object)
	for i, a in enumerate(names):
		for j, b in enumerate(names):
			both = df[a].notna() & df[b].notna()
			if i == j:
				out.iloc[i, j] = "-"
			elif i < j:
				out.iloc[i, j] = f"{(df.loc[both, a] == df.loc[both, b]).mean():.3f}"
			else:
				out.iloc[i, j] = f"k={cohen_kappa_score(df.loc[both, a], df.loc[both, b]):.3f}"
	return out


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=Path, default=OUTPUT, help=f"predictions CSV to score (default: {OUTPUT.name})")
	parser.add_argument("--split", choices=["all", "train", "heldout"], default="all")
	parser.add_argument("--models", nargs="*", help="default: every *_eligibility column in the output CSV")
	parser.add_argument("--confusion", action="store_true", help="print confusion matrices")
	parser.add_argument("--out", help="also write the summary table to this CSV")
	parser.add_argument("--answered", action="store_true", help="only score rows every selected model has answered (e.g. after a --limit smoke test)")
	parser.add_argument("--rows", help="write a per-criterion side-by-side comparison to this CSV")
	args = parser.parse_args()

	df = split_rows(load(args.input), args.split)
	available = model_columns(df)
	models = args.models or available
	for m in models:
		if m not in available:
			raise SystemExit(f"no {m}_eligibility column in {args.input.name}; available: {available}")
	if "jev" not in available:
		print("note: jev not run yet (python trial-jev/jev_eligibility.py)")
	if args.answered:
		df = df[df[[f"{m}_eligibility" for m in models]].notna().all(axis=1)]
		if df.empty:
			raise SystemExit("no rows answered by every selected model")

	if args.rows:
		cols = ["annotation_id", "patient_id", "trial_id", "criterion_type", "criterion_text", TRUTH]
		rows = df[cols].copy()
		for m in models:
			rows[m] = df[f"{m}_eligibility"]
			if f"{m}_confidence" in df and df[f"{m}_confidence"].notna().any():
				rows[f"{m}_confidence"] = df[f"{m}_confidence"].round(4)
			for label in ALL_LABELS:
				if prob_column(m, label) in df:
					rows[prob_column(m, label)] = df[prob_column(m, label)].round(4)
			rows[f"{m}_correct"] = df[f"{m}_eligibility"] == df[TRUTH]
		rows.to_csv(args.rows, index=False)
		print(f"wrote {len(rows)} rows to {args.rows}")

	n_type = df["criterion_type"].value_counts()
	print(f"\nsplit={args.split}: {len(df)} criteria ({n_type.get('inclusion', 0)} inclusion, {n_type.get('exclusion', 0)} exclusion)")
	print("reference: TrialGPT paper, GPT-4 acc 0.873 / inc 0.899 / exc 0.859; experts 0.887-0.900")

	table = pd.DataFrame([summary(df, m) for m in models]).set_index("model")
	print("\n" + table.to_string(float_format="%.4f"))

	for m in models:
		print(f"\n{m} per-class")
		print(per_class(df, m).to_string(index=False, float_format="%.4f"))
		calibration(df, m)
		option_probabilities(df, m)
		if args.confusion:
			confusion(df, m)

	print("\nagreement (upper: % same label, lower: Cohen's kappa)")
	print(agreement(df, models).to_string())

	if args.out:
		table.to_csv(args.out)
		print(f"\nwrote {args.out}")
