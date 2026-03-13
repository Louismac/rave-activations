import numpy as np
from pathlib import Path
import gin
import pickle
from rave import RAVE

from rave_activation_clustering import (
    RAVEActivationAnalyser
)

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
    device: str = 'mps',
):
    # 1. Load RAVE model
    print("\n" + "="*60)
    print("Step 1: Loading RAVE model")
    print("="*60)
    model = load_rave_model(model_path, config_path, device)
    
    # 2. Initialize analyser
    analyser = RAVEActivationAnalyser(model, device)
    
    return analyser


def load_dataset_cache(cache_path):
    """Load audio and metadata from cache file."""
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None, None

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    print(f"   Loaded cached dataset from {cache_path}")
    return cache_data['audio_list'], cache_data['metadata_list']


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


def run_analysis(analyser, audio_list, metadata_list, output_dir, n_clusters, pca_components):
    """Run clustering and correlation analysis."""
    # Convert channels to match model
    
    audio_list = convert_channels(audio_list, analyser.model.n_channels)

    analyser.activate(audio_list, metadata_list)
    #Within layer clustering not really used (layers are clusters)
    analyser.do_clustering(
        output_dir=output_dir,
        n_clusters=n_clusters,
        pca_components=pca_components
    )
    analyser.do_cross_layer_clustering(
        output_dir=output_dir,
        n_clusters=n_clusters,
        pca_components=pca_components
    )
    analyser.do_correlation(output_dir)
    analyser.do_cross_layer_correlation(
        output_dir=output_dir
    )
    with open(output_dir/ 'variance_correlation.json', 'r') as f:
        variance_data = json.load(f)

    with open(output_dir/ 'clustering_results.json', 'r') as f:
        clustering_data = json.load(f)

    # Tables
    create_summary_table(variance_data, clustering_data, output_dir)    

def create_summary_table(variance_data, clustering_data, output_dir):
    data_rows = []
    
    for layer in [l for l in variance_data.keys() if "net" in l]:
        row = {'Layer': layer}
        # Silhouette score
        if layer in clustering_data:
            row['Silhouette'] = f"{clustering_data[layer]['silhouette_score']:.3f}"
        else:
            row['Silhouette'] = 'N/A'
        
        # Pitch (Hz) stats
        if 'pitch' in variance_data[layer]:
            pitch_hz = variance_data[layer]['pitch']
            responsive = pitch_hz["n_responsive_neurons"]
            total = len(pitch_hz["all_correlations"])
            percent = (responsive/total)*100
            row['Pitch Hz Resp %'] = f"{percent:.1f}"
            row['Pitch Hz Corr'] = f"{pitch_hz['mean_correlation']:.3f}"
        else:
            row['Pitch Hz Resp %'] = '0.0'
            row['Pitch Hz Corr'] = '0.000'

        # Pitch class stats
        if 'pitch_class' in variance_data[layer]:
            pitch_class = variance_data[layer]['pitch_class']
            responsive = pitch_class["n_responsive_neurons"]
            total = len(pitch_class["all_correlations"])
            percent = (responsive/total)*100
            row['Pitch Class Resp %'] = f"{percent:.1f}"
            row['Pitch Class Corr'] = f"{pitch_class['mean_correlation']:.3f}"
        else:
            row['Pitch Class Resp %'] = '0.0'
            row['Pitch Class Corr'] = '0.000'

        # BPM stats
        if 'bpm' in variance_data[layer]:
            bpm = variance_data[layer]['bpm']
            responsive = bpm["n_responsive_neurons"]
            total = len(bpm["all_correlations"])
            percent = (responsive/total)*100
            row['BPM Resp %'] = f"{percent:.1f}"
            row['BPM Corr'] = f"{bpm['mean_correlation']:.3f}"
        else:
            row['BPM Resp %'] = '0.0'
            row['BPM Corr'] = '0.000'
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Sort by layer
    layer_nums = []
    for layer in df['Layer']:
        if 'net.' in layer:
            parts = layer.split('.')
            try:
                num = int(parts[1])
                layer_nums.append(num)
            except:
                layer_nums.append(999)
        else:
            layer_nums.append(999)
    
    df['sort_key'] = layer_nums
    df = df.sort_values('sort_key').drop('sort_key', axis=1)
    
    print("\n" + "="*80)
    print("Table 1: Layer-Level Statistics")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_dir = output_dir / 'table1_layer_statistics.csv'
    df.to_csv(output_dir, index=False)
    print("\nSaved to: table1_layer_statistics.csv")
    
    return df


def load_datasets(home):
    # Create cache directory
    cache_dir = home / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Create datasets once with 2 channels
    print("\n" + "="*70)
    print("Creating datasets (2 channels)")
    print("="*70)

    # Strings dataset
    print("\n1. Loading strings dataset...")
    strings_cache = cache_dir / "strings_dataset.pkl"
    strings_audio, strings_metadata = load_dataset_cache(strings_cache)

    if strings_audio is None:
        print("   Cache not found, creating dataset...")
        strings_audio, strings_metadata = AudioDataset.select_strings(
            "/home/louis/Documents/datasets/string_loops", 100, channels=2)
        print(f"   Loaded {len(strings_audio)} strings samples")

        # Apply pitch segment extraction to strings
        print("   Extracting pitched segments from strings...")
        strings_audio, strings_metadata = AudioDataset.extract_pitched_segments(
            strings_audio, strings_metadata,
            segment_duration=0.1,
            sample_rate=48000,
            fmin=80.0,
            fmax=800.0
        )
        print(strings_metadata)
        print(f"   Total: {len(strings_audio)} samples (original + pitched segments)")

        # Save to cache
        save_dataset_cache(strings_cache, strings_audio, strings_metadata)
    else:
        print(f"   Loaded {len(strings_audio)} samples from cache")

    # Drums dataset
    print("\n2. Loading drums dataset...")
    drums_cache = cache_dir / "drums_dataset.pkl"
    drums_audio, drums_metadata = load_dataset_cache(drums_cache)

    if drums_audio is None:
        print("   Cache not found, creating dataset...")
        drums_audio, drums_metadata = AudioDataset.select_drums(
            "/home/louis/Documents/datasets/drum_loops", 100, channels=2)
        print(f"   Loaded {len(drums_audio)} drum samples")

        # Apply pitch segment extraction to drums
        print("   Extracting pitched segments from drums...")
        drums_audio, drums_metadata = AudioDataset.extract_pitched_segments(
            drums_audio, drums_metadata,
            segment_duration=0.1,
            sample_rate=48000,
            fmin=80.0,
            fmax=800.0
        )
        print(drums_metadata)
        print(f"   Total: {len(drums_audio)} samples (original + pitched segments)")

        # Save to cache
        save_dataset_cache(drums_cache, drums_audio, drums_metadata)
    else:
        print(f"   Loaded {len(drums_audio)} samples from cache")

    # Stimuli dataset
    print("\n3. Generating stimuli dataset...")
    stimuli_cache = cache_dir / "stimuli_dataset.pkl"
    stimuli_audio, stimuli_metadata = load_dataset_cache(stimuli_cache)

    if stimuli_audio is None:
        print("   Cache not found, creating dataset...")
        stimuli_audio, stimuli_metadata = AudioDataset.generate_test_dataset(channels=2)
        print(f"   Generated {len(stimuli_audio)} stimuli samples")

        # Apply pitch segment extraction to stimuli
        print("   Extracting pitched segments from stimuli...")
        stimuli_audio, stimuli_metadata = AudioDataset.extract_pitched_segments(
            stimuli_audio, stimuli_metadata,
            segment_duration=0.1,
            sample_rate=48000,
            fmin=80.0,
            fmax=800.0
        )
        print(stimuli_metadata)
        print(f"   Total: {len(stimuli_audio)} samples (original + pitched segments)")

        # Save to cache
        save_dataset_cache(stimuli_cache, stimuli_audio, stimuli_metadata)
    else:
        print(f"   Loaded {len(stimuli_audio)} samples from cache")

    # Vocals dataset
    print("\n4. Loading vocals dataset...")
    vocals_cache = cache_dir / "vocals_dataset.pkl"
    vocals_audio, vocals_metadata = load_dataset_cache(vocals_cache)

    if vocals_audio is None:
        print("   Cache not found, creating dataset...")
        vocals_audio, vocals_metadata = AudioDataset.select_vocals_from_csv(
            "/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/taylor_extras/dataset_analysis",
            100, channels=2)
        print(f"   Loaded {len(vocals_audio)} vocal samples")

        # Apply pitch segment extraction to vocals
        print("   Extracting pitched segments from vocals...")
        vocals_audio, vocals_metadata = AudioDataset.extract_pitched_segments(
            vocals_audio, vocals_metadata,
            segment_duration=0.1,
            sample_rate=48000,
            fmin=80.0,
            fmax=800.0
        )
        print(vocals_metadata)
        print(f"   Total: {len(vocals_audio)} samples (original + pitched segments)")

        # Save to cache
        save_dataset_cache(vocals_cache, vocals_audio, vocals_metadata)
    else:
        print(f"   Loaded {len(vocals_audio)} samples from cache")

    print("\n" + "="*70)
    print("Datasets ready!")
    print("="*70)

    return [
        strings_audio, strings_metadata,
        drums_audio, drums_metadata,
        stimuli_audio, stimuli_metadata,
        vocals_audio, vocals_metadata
    ]

def run_cluster_analysis(home, datasets_dict, n_clusters = 6):    

    # Run analysis for each model
    for m in models:
        print(f"\n\n{'='*70}")
        print(f"Processing model: {m}")
        print(f"{'='*70}")

        model_path = home / "runs" / m / "best.ckpt"
        config_path = home / "runs" / m / "config.gin"
        output_dir = home / "results"  / f"{n_clusters}_cluster"/ m 

        # Strings analysis
        print(f"\n--- Analyzing strings dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["strings"][0], datasets_dict["strings"][1], output_dir / "strings", n_clusters, pca_components)

        # Drums analysis
        print(f"\n--- Analyzing drums dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["drum_loops"][0], datasets_dict["drum_loops"][1], output_dir / "drum_loops", n_clusters, pca_components)

        # Stimuli analysis
        print(f"\n--- Analyzing stimuli dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["stimuli"][0], datasets_dict["stimuli"][1], output_dir / "stimuli", n_clusters, pca_components)

        # Vocals analysis
        print(f"\n--- Analyzing vocals dataset ---")
        analyser = get_analyser(
            model_path=model_path,
            config_path=config_path,
            device="cuda"
        )
        run_analysis(analyser, datasets_dict["vocals"][0], datasets_dict["vocals"][1], output_dir / "vocals", n_clusters, pca_components)


if __name__ == "__main__":
    """
    Main execution block for running experiments
    """
    home = Path("/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/")
    models = ["strings", "drum_loops", "taylor_vocal"]
    datasets = load_datasets(home)
    pca_components = 2
    n_clusters = 4

    # Prepare datasets dictionary
    # Note: key names must match results folder names for cluster loading
    datasets_dict = {
        'strings': (datasets[0], datasets[1]),
        'drum_loops': (datasets[2], datasets[3]),
        'stimuli': (datasets[4], datasets[5]),
        'vocals': (datasets[6], datasets[7])
    }

    run_cluster_analysis(home, datasets_dict)
    