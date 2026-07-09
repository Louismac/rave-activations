import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
BALANCED_SAMPLE_SEED = 42

def stats_for_full_datasets():
    vocal_path = "/home/louis/Documents/datasets/taylor/separated/htdemucs"
    string_path = "/home/louis/Documents/datasets/string_loops"
    drum_path = "/home/louis/Documents/datasets/drum_loops"

    import soundfile as sf

    def dataset_stats(path):
        if not "taylor" in path:
            wavs = sorted(Path(path).rglob("*.wav"))
        else:
            wavs = sorted(Path(path).rglob("*vocals.wav"))
        durations = []
        for w in wavs:
            try:
                info = sf.info(str(w))
                durations.append(info.frames / info.samplerate)
            except:
                print(f"error reading {str(w)}")
        durations = np.array(durations)
        return {
            "num_files": len(durations),
            "total_duration_s": float(durations.sum()),
            "mean_duration_s": float(durations.mean()) if len(durations) else 0.0,
            "std_duration_s": float(durations.std()) if len(durations) else 0.0,
        }

    results = {
        "vocals": dataset_stats(vocal_path),
        "strings": dataset_stats(string_path),
        "drums": dataset_stats(drum_path),
    }

    for name, s in results.items():
        print(f"\n{name}:")
        print(f"  files:          {s['num_files']}")
        print(f"  total duration: {s['total_duration_s']:.1f} s  ({s['total_duration_s']/60:.1f} min)")
        print(f"  mean length:    {s['mean_duration_s']:.2f} s")
        print(f"  std length:     {s['std_duration_s']:.2f} s")

    return results

PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

import numpy as np
from typing import Optional


def balanced_sample(
    feature_values: np.ndarray,
    n: int,
    n_bins: int = 10,
    binning: str = "linear",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Select indices for a sample of size `n` balanced across the range of
    `feature_values`. Bins the value range into `n_bins` bins and samples
    approximately equal numbers from each bin.

    Parameters
    ----------
    feature_values : 1D array-like of float
        The feature values for all available samples. Indexed 0..len-1.
    n : int
        Target sample size.
    n_bins : int
        Number of bins to stratify across. Default 10 works well for N=500.
    binning : {"linear", "log", "quantile"}
        - "linear": equal-width bins across [min, max] of feature_values
        - "log": equal-width bins in log space (good for pitch, centroid)
        - "quantile": equal-count bins based on data quantiles (no balancing
                      benefit but useful as a control / fallback)
    rng : np.random.Generator, optional
        For reproducibility. If None, uses np.random.default_rng() with no seed.

    Returns
    -------
    indices : 1D np.ndarray of int
        Indices into feature_values, of length min(n, total available samples).
        Sorted in ascending order for consistency.

    Notes
    -----
    - If a bin has fewer samples than the target per-bin count, all samples
      in that bin are taken; remaining slots are filled from other bins.
    - If total available samples < n, returns all available indices.
    - Bins that contain no samples are skipped silently.
    """
    feature_values = np.asarray(feature_values, dtype=float)
    n_total = len(feature_values)

    if n_total <= n:
        return np.arange(n_total)

    if rng is None:
        rng = np.random.default_rng()

    # ── set up bin edges ──────────────────────────────────────────────
    fv_min, fv_max = feature_values.min(), feature_values.max()

    if binning == "linear":
        edges = np.linspace(fv_min, fv_max, n_bins + 1)
    elif binning == "log":
        if fv_min <= 0:
            raise ValueError("log binning requires all feature values > 0")
        edges = np.geomspace(fv_min, fv_max, n_bins + 1)
    elif binning == "quantile":
        edges = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"unknown binning: {binning}")

    # ── assign each sample to a bin ───────────────────────────────────
    # np.digitize with right=False maps x to bin i where edges[i-1] <= x < edges[i]
    # We want bins 0..n_bins-1, with the max value going into the last bin
    bin_ids = np.digitize(feature_values, edges[1:-1])  # values 0..n_bins-1

    # ── target sample count per bin ───────────────────────────────────
    target_per_bin = n // n_bins
    remainder = n - target_per_bin * n_bins  # distribute leftover

    # ── sample from each bin ──────────────────────────────────────────
    selected_indices = []
    deficit_bins = []  # bins that had fewer samples than target

    for b in range(n_bins):
        in_bin = np.where(bin_ids == b)[0]
        target_this_bin = target_per_bin + (1 if b < remainder else 0)

        if len(in_bin) >= target_this_bin:
            chosen = rng.choice(in_bin, size=target_this_bin, replace=False)
        else:
            # take all available, mark deficit
            chosen = in_bin
            deficit_bins.append((b, target_this_bin - len(in_bin)))

        selected_indices.extend(chosen.tolist())

    # ── fill deficits from remaining samples ──────────────────────────
    total_deficit = sum(d for _, d in deficit_bins)
    if total_deficit > 0:
        remaining = np.setdiff1d(np.arange(n_total), np.array(selected_indices))
        if len(remaining) > 0:
            n_extra = min(total_deficit, len(remaining))
            extra = rng.choice(remaining, size=n_extra, replace=False)
            selected_indices.extend(extra.tolist())

    return np.sort(np.array(selected_indices))


def get_balanced_indices(feature_values, n, feature_name, rng=None):
    """
    Convenience wrapper that picks the right binning scheme per feature.

    Pitch and spectral centroid are log-spaced (perceptually).
    BPM and spectral bandwidth are linear.
    """
    binning_map = {
        "pitch":              "log",
        "spectral_centroid":  "log",
        "bpm":                "linear",
        "spectral_bandwidth": "linear",
    }
    binning = binning_map.get(feature_name, "linear")
    return balanced_sample(feature_values, n=n, binning=binning, rng=rng)

def print_dataset_stats(name, balanced_data, plot_dir=None):
    """
    Print per-feature stats and save a distribution plot, reading from a
    pre-balanced dataset (see make_balanced_dataset.py) rather than balancing
    live. balanced_data is the dict loaded by load_balanced_cache(): one
    {'audio_list', 'metadata_list'} subset per feature, already balanced
    across that feature's value range and capped at 500, with bpm-pass /
    pitch-pass entries already correctly excluded from each other's pool.
    """
    features = ["bpm", "pitch", "spectral_centroid", "spectral_bandwidth"]
    total_loaded = len(set(
        m["original_index"]
        for feat in features if feat in balanced_data
        for m in balanced_data[feat]["metadata_list"]
    ))
    print(f"\n{'='*55}")
    print(f"  {name}  —  {total_loaded} balanced samples loaded")
    print(f"{'='*55}")

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle(f"{name} — feature distributions", fontsize=13)

    for ax, feat in zip(axes, features):
        metadata_list = balanced_data.get(feat, {}).get("metadata_list", [])

        raw = []
        for m in metadata_list:
            if feat in m and m[feat] is not None:
                try:
                    raw.append(float(m[feat]))
                except (ValueError, TypeError):
                    pass

        if not raw:
            ax.set_title(f"{feat}\n(no data)")
            ax.axis("off")
            continue

        arr = np.array(raw)

        print(f"\n  {feat}:")
        print(f"    n={len(arr)}   range=[{arr.min():.2f}, {arr.max():.2f}]"
              f"   mean={arr.mean():.2f}   std={arr.std():.2f}")

        if feat == "pitch_class":
            counts = np.bincount(arr.astype(int), minlength=12)
            ax.bar(range(12), counts, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(12))
            ax.set_xticklabels(PITCH_CLASS_NAMES, fontsize=8)
        else:
            ax.hist(arr, bins=25, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.set_xlabel(feat)

        ax.set_title(f"{feat}  (n={len(arr)})\nμ={arr.mean():.1f}  σ={arr.std():.1f}"
                     f"  [{arr.min():.1f}–{arr.max():.1f}]")
        ax.set_ylabel("count")

    plt.tight_layout()
    save_path = Path(plot_dir or ".") / f"{name}_feature_distributions.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved → {save_path}")


def load_dataset_cache(cache_path):
    """Load audio and metadata from cache file."""
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None, None

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    print(f"   Loaded cached dataset from {cache_path}")
    return cache_data['audio_list'], cache_data['metadata_list']


def load_balanced_cache(cache_path):
    """Load a *_balanced.pkl produced by make_balanced_dataset.py.

    Returns a dict keyed by feature name, each holding its own balanced
    {'audio_list', 'metadata_list'} subset, or None if the file doesn't exist.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    with open(cache_path, 'rb') as f:
        balanced_data = pickle.load(f)

    print(f"   Loaded balanced cache from {cache_path}")
    return balanced_data


def load_datasets():
    # Create cache directory
    cache_dir = home / "cache" / "500_pitch_100_bpm_4_sc_4_sb_4"
    cache_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("Loading balanced datasets")
    print("="*70)

    plots_dir = home / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    for i, name in enumerate(["strings", "drums", "stimuli", "vocals"], start=1):
        print(f"\n{i}. Loading {name} dataset...")
        balanced_data = load_balanced_cache(cache_dir / f"{name}_dataset_features_balanced.pkl")
        if balanced_data:
            print_dataset_stats(name, balanced_data, plot_dir=plots_dir)



if __name__ == "__main__":
    """
    Main execution block for running experiments
    """
    home = Path("/home/louis/Documents/notebooks/rave-activations/rave-activations/")
    # stats_for_full_datasets()
    datasets = load_datasets()
