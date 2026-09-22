"""Paired bootstrap confidence intervals for performance differences.

Quantifies uncertainty due to the finite test cohort (as opposed to training
stochasticity, which is captured by the mean +/- std over repeated runs).

For each dataset and each baseline, the per-image paired difference
    d_i = score(proposed, image i) - score(baseline, image i)
is resampled with replacement over IMAGES (B times). The reported interval is
the percentile bootstrap CI of the mean difference. A CI excluding zero
indicates the improvement is not explained by test-cohort sampling.

Input: a long-format CSV with one row per (dataset, model, image, metric value):

    dataset,model,image_id,dice,iou
    TN3K,GBKA-Net,tn3k_0001.png,0.9312,0.8712
    TN3K,U-KAN,tn3k_0001.png,0.9008,0.8195
    ...

All models must cover the same image_ids within a dataset; the script pairs on
image_id and refuses to proceed otherwise.

Usage:
    python bootstrap_ci.py --scores per_image_scores.csv \
        --proposed "GBKA-Net" --baselines "U-KAN,U-Mamba,U-Net" \
        --n_boot 10000 --seed 42 --latex ci_table.tex
"""
import argparse
import numpy as np
import pandas as pd


def paired_bootstrap(diff, n_boot, alpha, rng):
    """Percentile bootstrap CI for the mean of paired differences."""
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))       # resample IMAGES
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diff.mean(), lo, hi


def run(df, proposed, baselines, metrics, n_boot, alpha, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, sub in df.groupby("dataset", sort=False):
        piv = {m: g.set_index("image_id") for m, g in sub.groupby("model", sort=False)}
        if proposed not in piv:
            print(f"  [skip] {dataset}: no rows for '{proposed}'")
            continue
        for base in baselines:
            if base not in piv:
                continue
            a, b = piv[proposed], piv[base]
            common = a.index.intersection(b.index)
            if len(common) != len(a.index) or len(common) != len(b.index):
                raise ValueError(
                    f"{dataset}/{base}: image sets differ "
                    f"({len(a.index)} vs {len(b.index)}, {len(common)} shared). "
                    "Per-image scores must cover identical images.")
            entry = {"dataset": dataset, "baseline": base, "n": len(common)}
            for met in metrics:
                diff = (a.loc[common, met].to_numpy(float)
                        - b.loc[common, met].to_numpy(float))
                mean, lo, hi = paired_bootstrap(diff, n_boot, alpha, rng)
                entry[f"{met}_mean"] = mean
                entry[f"{met}_lo"] = lo
                entry[f"{met}_hi"] = hi
                entry[f"{met}_excl0"] = (lo > 0) or (hi < 0)
            rows.append(entry)
    return pd.DataFrame(rows)


PRETTY = {"dice": "Dice", "iou": "IoU", "hd95": "HD95"}


def to_latex(res, metrics, proposed, conf, scale):
    unit = r" (\%)" if scale == 100 else ""
    head = " & ".join([r"\textbf{Dataset}", r"\textbf{Comparison}", r"\textbf{$N$}"]
                      + [rf"\textbf{{$\Delta$ {PRETTY.get(m, m)}{unit}}}"
                         for m in metrics])
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Paired bootstrap {conf:.0f}\% confidence intervals for the "
        rf"difference in performance between {proposed} and each baseline on the "
        r"fixed test sets. Intervals are obtained by resampling test images with "
        r"replacement; an interval excluding zero indicates that the improvement "
        r"is not attributable to test-cohort sampling.}",
        r"\label{tab:bootstrap}", r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{ll" + "c" * (1 + len(metrics)) + "}", r"\hline",
        head + r" \\", r"\hline",
    ]
    for _, r in res.iterrows():
        cells = [str(r["dataset"]), f"vs.\\ {r['baseline']}", str(int(r["n"]))]
        for m in metrics:
            cells.append(f"{r[f'{m}_mean']*scale:.2f} "
                         f"[{r[f'{m}_lo']*scale:.2f}, {r[f'{m}_hi']*scale:.2f}]")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="long-format CSV of per-image scores")
    ap.add_argument("--proposed", default="GBKA-Net")
    ap.add_argument("--baselines", default="U-KAN",
                    help="comma-separated baseline model names")
    ap.add_argument("--metrics", default="dice,iou")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 -> 95%% CI")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=100.0,
                    help="100 if scores are 0-1 and you report percentages")
    ap.add_argument("--latex", default=None, help="optional path for a LaTeX table")
    ap.add_argument("--csv_out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    metrics = [m.strip() for m in args.metrics.split(",")]
    baselines = [b.strip() for b in args.baselines.split(",")]
    for col in ["dataset", "model", "image_id"] + metrics:
        if col not in df.columns:
            raise ValueError(f"missing column '{col}' in {args.scores}")

    res = run(df, args.proposed, baselines, metrics, args.n_boot, args.alpha, args.seed)
    if res.empty:
        raise SystemExit("No comparisons produced; check --proposed / --baselines names.")

    conf = 100 * (1 - args.alpha)
    print(f"\nPaired bootstrap {conf:.0f}% CIs "
          f"({args.n_boot} resamples, seed {args.seed}) for {args.proposed} - baseline\n")
    for _, r in res.iterrows():
        parts = []
        for m in metrics:
            star = "" if r[f"{m}_excl0"] else "   (includes 0)"
            parts.append(f"{m.upper()} {r[f'{m}_mean']*args.scale:+.2f} "
                         f"[{r[f'{m}_lo']*args.scale:+.2f}, "
                         f"{r[f'{m}_hi']*args.scale:+.2f}]{star}")
        print(f"  {r['dataset']:<8} vs {r['baseline']:<10} (N={int(r['n'])})  "
              + " | ".join(parts))

    if args.csv_out:
        res.to_csv(args.csv_out, index=False)
        print(f"\nsaved {args.csv_out}")
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(to_latex(res, metrics, args.proposed, conf, args.scale))
        print(f"saved {args.latex}")


if __name__ == "__main__":
    main()
