#!/usr/bin/env bash
set -uo pipefail

project_root="/Users/josephdaher/Git Repositories/djprojectexploration"
mix_dir="/Users/josephdaher/Git Repositories/ytdlp/mixes"

mixes=(
  helenahauff2017dekmantel
  jeffmills1996mixupvol2
  kaytranada2013boilerroommontreal
  kilimanjaro2025mixmaglab
  majorleaguedjzdbngogo2021balconymix
  mala2015boilerroomlondon
  motorcitydrumensemble2014dekmantel
  nooriyah2022boilerroommiddleofnowhere
  prydz2025toronto
  rosapistola2017boilerroomnewyork
  sherelle2019boilerroomfestival

  # Uncomment after fixing/reviewing their cue sheets:
  # blackcoffee2018cercle
  # yungsingh2021boilerroomdaytimers
)

cd "$project_root" || exit 1

failed=()
successful=()

for slug in "${mixes[@]}"; do
  audio="$mix_dir/$slug.mp3"
  cues="$mix_dir/$slug.csv"
  manifest="$project_root/data/exports/${slug}_rendered_set_manifest.json"
  metrics="$project_root/data/mix_evaluation/${slug}_simplex_metrics.csv"

  echo
  echo "===== $slug ====="

  if [[ ! -f "$audio" || ! -f "$cues" ]]; then
    echo "Missing audio or cue CSV; skipping $slug" >&2
    failed+=("$slug:inputs")
    continue
  fi

  if [[ -f "$manifest" ]]; then
    echo "Feature manifest already exists; skipping extraction."
  else
    if ! uv run djprojectexploration-rendered-set \
      "$audio" \
      "$cues" \
      --analysis-guards \
      --analysis-head-guard-sec 20 \
      --analysis-tail-guard-sec 30 \
      --min-analysis-duration-sec 30
    then
      echo "Feature extraction failed for $slug" >&2
      failed+=("$slug:features")
      continue
    fi
  fi

  if [[ -f "$metrics" ]]; then
    echo "Evaluation output already exists; skipping evaluation."
  else
    if ! uv run djprojectexploration-evaluate-mix-set \
      "$manifest" \
      --output-dir "$project_root/data/mix_evaluation" \
      --simplex-step 0.10 \
      --permutations 500000 \
      --random-seed 20240722 \
      --rhythm-tempo-weight 0.70
    then
      echo "Evaluation failed for $slug" >&2
      failed+=("$slug:evaluation")
      continue
    fi
  fi

  successful+=("$slug")
done

if ((${#failed[@]} > 0)); then
  echo
  echo "Some mixes failed:"
  printf '  %s\n' "${failed[@]}"
  echo "The completed stages will be skipped when this script is rerun."
  exit 1
fi

# Replace rows for these mixes in the combined dataset, preserving all other
# previously evaluated mixes.
export DJ_SUCCESSFUL_MIX_SLUGS="${successful[*]}"

uv run python - <<'PY'
import os
from pathlib import Path

import pandas as pd

project_root = Path("/Users/josephdaher/Git Repositories/djprojectexploration")
evaluation_dir = project_root / "data" / "mix_evaluation"
combined_path = evaluation_dir / "all_processed_simplex_metrics.csv"
slugs = os.environ["DJ_SUCCESSFUL_MIX_SLUGS"].split()

new_frames = [
    pd.read_csv(evaluation_dir / f"{slug}_simplex_metrics.csv")
    for slug in slugs
]
new_metrics = pd.concat(new_frames, ignore_index=True)

if combined_path.exists():
    existing = pd.read_csv(combined_path)

    if list(existing.columns) != list(new_metrics.columns):
        raise RuntimeError(
            "Combined and new metric columns differ; refusing to overwrite."
        )

    existing = existing.loc[~existing["mix_slug"].isin(slugs)]
    combined = pd.concat([existing, new_metrics], ignore_index=True)
else:
    combined = new_metrics

temporary_path = combined_path.with_suffix(".csv.tmp")
combined.to_csv(temporary_path, index=False)
temporary_path.replace(combined_path)

print(
    f"Wrote {combined_path} with "
    f"{combined['mix_slug'].nunique()} mixes and {len(combined)} metric rows."
)
PY

echo
echo "All requested mixes completed successfully."
