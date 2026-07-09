import numpy as np
from pathlib import Path
import gin
import pickle
from rave import RAVE
import json
import pandas as pd
from encodec_adapter import EncodecActivationAnalyser, load_encodec


from rave_activation_clustering import (
    RAVEActivationAnalyser
)

PROP_LABELS = {
    'pitch':             'Pitch Hz',
    'bpm':               'BPM',
    'spectral_centroid': 'SC',
    'spectral_bandwidth': 'SB',
}

def load_rave_model(model_path: str, config_path: str, device: str = 'cuda'):
    """
    Load a pretrained RAVE model
    
    Args:
        model_path: Path to RAVE model (.ts, .ckpt, or .gin file)
        device: 'cuda' or 'cpu'
    """
    from pathlib import Path
    model_path = Path(model_path)
    
    print("\n" + "-"*70)
    print("Step 1: Loading with RAVE library")
    print("-"*70)

        
    print(f"\nLoading checkpoint...")
    gin.clear_config()
    gin.parse_config_file(config_path)

    model = RAVE.load_from_checkpoint(
        model_path,
        strict=False
        ).eval()
    print("✓ Checkpoint loaded successfully!")
    

    return model
    
def get_analyser(
    model_path: str,
    config_path: str,
    device: str = 'cuda',
):
    # 1. Load RAVE model
    print("\n" + "="*60)
    print("Step 1: Loading RAVE model")
    print("="*60)
    model = load_rave_model(model_path, config_path, device)
    
    # 2. Initialize analyser
    analyser = RAVEActivationAnalyser(model, device)
    
    return analyser


def load_balanced_cache(cache_path):
    """Load a *_balanced.pkl produced by dataset/make_balanced_dataset.py.

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


def merge_balanced_dataset(balanced_data, activation_features=None):
    """
    Concatenate every feature's own balanced subset as-is — each already
    carries the audio actually appropriate for that feature (short
    pitch-tracked segment for 'pitch'; full chunk_duration window for
    'bpm'/'spectral_centroid'/'spectral_bandwidth' — see
    dataset/make_balanced_dataset.py). No cross-feature deduplication by
    original_index: the same underlying sample can be selected into more
    than one feature's balanced subset, and when it is, each occurrence is
    kept with that feature's own (possibly different-duration) audio rather
    than collapsed to a single row. This trades some duplicated audio/compute
    for a simple, unambiguous mapping — feature_indices[feature] is exactly
    the set of merged-list positions built from that feature's subset, no
    original_index cross-referencing required downstream.
    """
    if activation_features is None:
        activation_features = tuple(balanced_data.keys())

    audio_list, metadata_list = [], []
    feature_indices = {}

    for feature in activation_features:
        subset = balanced_data.get(feature)
        if subset is None:
            continue
        positions = set()
        for audio, meta in zip(subset['audio_list'], subset['metadata_list']):
            positions.add(len(metadata_list))
            audio_list.append(audio)
            metadata_list.append(meta)
        feature_indices[feature] = positions

    return audio_list, metadata_list, feature_indices


def load_balanced_datasets(cache_dir):
    """
    Load each dataset's *_balanced.pkl (see dataset/make_balanced_dataset.py),
    concatenate its per-feature subsets, and return a dict keyed the same way
    as the datasets_dict built from load_datasets(), but with each value
    holding 'audio_list' / 'metadata_list' (concatenated, duplicates allowed)
    and 'feature_indices' (per-feature set of merged-list positions to pull
    activations from).
    """
    cache_dir = Path(cache_dir)
    cache_filenames = {
        'strings':    'strings_dataset_features_balanced_44100.pkl',
        'drum_loops': 'drums_dataset_features_balanced_44100.pkl',
        'stimuli':    'stimuli_dataset_features_balanced_44100.pkl',
        'vocals':     'vocals_dataset_features_balanced_44100.pkl',
    }

    datasets_dict = {}
    for name, filename in cache_filenames.items():
        balanced_data = load_balanced_cache(cache_dir / filename)
        if balanced_data is None:
            print(f"   No balanced cache found for {name} at {cache_dir / filename}, skipping.")
            continue

        audio_list, metadata_list, feature_indices = merge_balanced_dataset(balanced_data)
        sizes = ', '.join(f"{k}={len(v)}" for k, v in feature_indices.items())
        print(f"   {name}: merged {len(metadata_list)} balanced samples ({sizes})")

        datasets_dict[name] = {
            'audio_list': audio_list,
            'metadata_list': metadata_list,
            'feature_indices': feature_indices,
        }

    return datasets_dict


def convert_channels(audio_list, target_channels):
    """Convert audio list to target number of channels."""
    if target_channels == 1:
        # Convert stereo to mono by averaging channels
        return [a.mean(dim=0, keepdim=True) if a.dim() > 1 and a.shape[0] > 1 else a for a in audio_list]
    elif target_channels == 2:
        # Convert mono to stereo by duplicating channel
        return [a.repeat(2, 1) if a.dim() == 1 or a.shape[0] == 1 else a for a in audio_list]
    else:
        return audio_list

def run_analysis(analyser, audio_list, metadata_list, output_dir, n_clusters, pca_components, model,
                  feature_indices=None, update_props=None, recluster=False):
    """
    Run clustering and correlation analysis.

    update_props: None (default) recomputes every property in PROP_LABELS and
        fully overwrites variance_correlation.json /
        cross_layer_cluster_correlation_all_neurons.json. Pass a subset (e.g.
        ['spectral_centroid', 'spectral_bandwidth']) to recompute only those
        properties and merge them into the existing correlation files,
        leaving other properties (pitch/bpm/...) exactly as they were.
    recluster: if False (default), do_clustering/do_cross_layer_clustering are
        skipped when their output JSON already exists in output_dir —
        clustering doesn't depend on which property is being correlated, so
        there's no need to redo it just to refresh a subset of properties.
        Pass True to force recomputing clusters too (e.g. after a change that
        affects the activation population itself, not just one property).
    """
    output_dir = Path(output_dir)

    # Convert channels to match model
    chans = 1
    if hasattr(analyser.model, "n_channels"):
        chans = analyser.model.n_channels
    audio_list = convert_channels(audio_list, chans)

    analyser.activate(audio_list, metadata_list)
    if feature_indices is not None:
        analyser.set_balanced_feature_indices(feature_indices)

    cross_layer_path = output_dir / "cross_layer_clustering_results_all_neurons.json"

    if recluster or not cross_layer_path.exists():
        analyser.do_cross_layer_clustering(
            output_dir=output_dir,
            n_clusters=n_clusters,
            pca_components=pca_components
        )
    else:
        print(f"  {cross_layer_path.name} already exists, skipping do_cross_layer_clustering "
              f"(pass recluster=True to force).")

    prop = update_props if update_props is not None else list(PROP_LABELS.keys())
    update = update_props is not None

    analyser.do_correlation(output_dir, prop=prop, update=update)
    analyser.do_cross_layer_correlation(output_dir=output_dir, prop=prop, update=update)
    with open(output_dir/ 'variance_correlation.json', 'r') as f:
        variance_data = json.load(f)

    # Tables
    create_summary_table(variance_data, output_dir)

def create_summary_table(variance_data, output_dir,
                         properties=None):
    if properties is None:
        properties = list(PROP_LABELS.keys())

    data_rows = []
    EXCLUDE_PREFIXES = ("section_",)
    layer_keys = [l for l in variance_data.keys()
                  if not l.startswith(EXCLUDE_PREFIXES)]

    for layer in layer_keys:
        row = {'Layer': layer}
        for prop in properties:
            label = PROP_LABELS.get(prop, prop)
            if prop in variance_data[layer]:
                d = variance_data[layer][prop]
                total = len(d["all_correlations"])
                row[f'{label} Resp %'] = f"{(d['n_responsive_neurons']/total)*100:.1f}"
                row[f'{label} Corr']   = f"{d['mean_correlation']:.3f}"
            else:
                row[f'{label} Resp %'] = '0.0'
                row[f'{label} Corr']   = '0.000'
        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Sort by depth. Both RAVE ('net.X.*') and EnCodec ('layers.X.*') carry the
    # depth index as the second dot-separated token; extract it generically.
    def _layer_sort_key(layer_name):
        parts = str(layer_name).split('.')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 999

    df['sort_key'] = df['Layer'].map(_layer_sort_key)
    df = df.sort_values('sort_key', kind='stable').drop('sort_key', axis=1)

    output_path = output_dir / 'table1_layer_statistics.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return df


def run_encodec_analysis(home, results_dir, datasets_dict,
                         n_clusters = 6, pca_components = 2,
                         update_props=None, recluster=False):
    model = load_encodec("facebook/encodec_32khz")        # music-oriented
    analyser = EncodecActivationAnalyser(model, device="cuda")
    m = "encodec"
    output_dir = home / results_dir  / f"{n_clusters}_cluster"/ "encodec"
    # structure = analyser.print_decoder_structure(sample_rate=32000)

    run_analysis(analyser, datasets_dict["strings"]["audio_list"],
                    datasets_dict["strings"]["metadata_list"], output_dir / "strings",
                    n_clusters, pca_components, m,
                    feature_indices=datasets_dict["strings"]["feature_indices"],
                    update_props=update_props, recluster=recluster)

    analyser = EncodecActivationAnalyser(model, device="cuda")
    run_analysis(analyser, datasets_dict["drum_loops"]["audio_list"],
                     datasets_dict["drum_loops"]["metadata_list"], output_dir / "drum_loops",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["drum_loops"]["feature_indices"],
                     update_props=update_props, recluster=recluster)

    analyser = EncodecActivationAnalyser(model, device="cuda")
    run_analysis(analyser, datasets_dict["stimuli"]["audio_list"],
                     datasets_dict["stimuli"]["metadata_list"], output_dir / "stimuli",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["stimuli"]["feature_indices"],
                     update_props=update_props, recluster=recluster)
    analyser = EncodecActivationAnalyser(model, device="cuda")
    run_analysis(analyser, datasets_dict["vocals"]["audio_list"],
                     datasets_dict["vocals"]["metadata_list"], output_dir / "vocals",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["vocals"]["feature_indices"],
                     update_props=update_props, recluster=recluster)

def run_cluster_analysis(home, results_dir, datasets_dict,
                         n_clusters = 6, pca_components = 2,
                         update_props=None, recluster=False):

    # Run analysis for each model
    for m in models:
        print(f"\n\n{'='*70}")
        print(f"Processing model: {m}")
        print(f"{'='*70}")

        model_path = home / "runs" / m / "best.ckpt"
        config_path = home / "runs" / m / "config.gin"
        output_dir = home / results_dir  / f"{n_clusters}_cluster"/ m

        # Strings analysis
        print(f"\n--- Analyzing strings dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["strings"]["audio_list"],
                     datasets_dict["strings"]["metadata_list"], output_dir / "strings",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["strings"]["feature_indices"],
                     update_props=update_props, recluster=recluster)

        # Drums analysis
        print(f"\n--- Analyzing drums dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["drum_loops"]["audio_list"],
                     datasets_dict["drum_loops"]["metadata_list"], output_dir / "drum_loops",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["drum_loops"]["feature_indices"],
                     update_props=update_props, recluster=recluster)

        # Stimuli analysis
        print(f"\n--- Analyzing stimuli dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["stimuli"]["audio_list"],
                     datasets_dict["stimuli"]["metadata_list"], output_dir / "stimuli",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["stimuli"]["feature_indices"],
                     update_props=update_props, recluster=recluster)

        # Vocals analysis
        print(f"\n--- Analyzing vocals dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["vocals"]["audio_list"],
                     datasets_dict["vocals"]["metadata_list"], output_dir / "vocals",
                     n_clusters, pca_components, m,
                     feature_indices=datasets_dict["vocals"]["feature_indices"],
                     update_props=update_props, recluster=recluster)


if __name__ == "__main__":
    """
    Main execution block for running experiments
    """
    import pathlib
    home = pathlib.Path(__file__).parent.resolve()
    models = ["strings", "drum_loops", "taylor_vocal"]
    cache_dir = home / "cache" / "500_pitch_100_bpm_4_sc_4_sb_4"
    datasets_dict = load_balanced_datasets(home / "cache" / "500_pitch_100_bpm_4_sc_4_sb_4")
    pca_components = 2
    #K_RANGE  = [4,5,7,8,9,10]
    K_RANGE = [6]
    for k in K_RANGE:
        run_cluster_analysis(home, "results_44100" , 
                     datasets_dict, n_clusters=k, pca_components = pca_components,
                     recluster=True)
        run_encodec_analysis(home, "results_44100" , 
                     datasets_dict, n_clusters=k, pca_components = pca_components,
                     recluster=True)
    