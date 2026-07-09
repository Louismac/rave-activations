

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import spearmanr
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent / "dataset"))
from dataset_stats import get_balanced_indices

# Fixed seed for balanced-sample selection so the same 500 indices per feature
# are drawn across all baseline scripts (each runs in its own process/seed state).
BALANCED_SAMPLE_SEED = 42


@dataclass
class ActivationRecord:
    """Store activations from a specific layer"""
    layer_name: str
    activations: np.ndarray  # Shape: (n_samples, n_neurons)
    
    
@dataclass
class ClusterResult:
    """Store clustering results"""
    cluster_labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    cluster_centers: Optional[np.ndarray] = None
    method: Optional[str] = None
    pca_components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None


class ActivationHook:
    """PyTorch hook to capture layer activations"""
    
    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.activations = []
        self.metadata = []
        
    def __call__(self, module, input, output):
        """Hook function called during forward pass"""
        # Store activation as numpy array
        # Handle different output types (tensor, tuple, etc.)
        if isinstance(output, torch.Tensor):
            act = output.detach().cpu().numpy()
        elif isinstance(output, tuple):
            act = output[0].detach().cpu().numpy()
        else:
            return
            
        # Flatten spatial dimensions, keep batch and channel dims
        # Shape: (batch, channels, *spatial) -> (batch, channels)
        if len(act.shape) > 2:
            # Average pool over spatial dimensions
            act = act.mean(axis=tuple(range(2, len(act.shape))))
        
        self.activations.append(act)
        
    def reset(self):
        """Clear stored activations"""
        self.activations = []
        self.metadata = []
        
    def get_activations(self) -> np.ndarray:
        """Return concatenated activations"""
        if not self.activations:
            return np.array([])
        return np.concatenate(self.activations, axis=0)

class RAVEActivationAnalyser:
    """
    Main class for RAVE activation analysis and clustering
    """
    
    def __init__(self, rave_model: nn.Module, device: str = 'cuda'):
        """
        Args:
            rave_model: Trained RAVE model
            device: 'cuda' or 'cpu'
        """
        self.model = rave_model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Storage for hooks and activations
        self.hooks = {}
        self.hook_handles = []
        self.activation_records = {}
        
        # Clustering results
        self.cluster_results = {}
        self.correlations = {}
        self.pca_models = {}

    def print_decoder_structure(self, dummy_seconds: float = 1.0,
                            sample_rate: int = 32000,
                            batch: int = 2):
        """
        Register decoder hooks, run a single dummy batch so every hook fires,
        and print the resulting layer structure: ordinal index, hooked name,
        module type, and captured output shape (post spatial-mean-pool, i.e.
        the (batch, n_neurons) shape the analysis actually uses).

        Call this BEFORE a full run to inspect layer naming and work out the
        early/middle/late section boundaries for a new architecture.

        Returns the list of (name, module_type, n_neurons) for programmatic use.
        """
        # Ensure hooks exist
        if len(self.hooks) == 0:
            self.register_decoder_hooks()
        if len(self.hooks) == 0:
            self.register_decoder_hooks(hook_all_leaf_modules=True)
        if len(self.hooks) == 0:
            raise RuntimeError("No hooks registered; cannot inspect structure.")

        for hook in self.hooks.values():
            hook.reset()

        # One dummy mono batch at the model's expected rate
        n_samples = int(dummy_seconds * sample_rate)
        dummy = torch.randn(batch, 1, n_samples).to(self.device)

        # Forward pass — use the same path the analyser uses for this model.
        # EncodecActivationAnalyser overrides _encodec_forward; RAVE uses encode/decode.
        with torch.no_grad():
            if hasattr(self, "_encodec_forward"):
                self._encodec_forward(dummy)
            elif hasattr(self.model, "decode"):
                z = self.model.encode(dummy)
                if isinstance(z, torch.Tensor):
                    z = z[:, :128, :]
                self.model.decode(z)
            else:
                self.model(dummy)

        # The hooks fired in module-execution order. Recover that order from the
        # decoder's named_modules() so the printout matches forward-pass depth.
        decoder = self._find_decoder()
        exec_order = [name for name, _ in decoder.named_modules() if name in self.hooks]

        print("\n" + "=" * 78)
        print(f"DECODER STRUCTURE  ({type(self.model).__name__})")
        print(f"{len(self.hooks)} hooked layers")
        print("=" * 78)
        print(f"{'idx':>4}  {'layer name':<46} {'type':<18} {'n_neurons':>9}")
        print("-" * 78)

        structure = []
        for idx, name in enumerate(exec_order):
            hook = self.hooks[name]
            acts = hook.get_activations()
            if len(acts) == 0:
                n_neurons = 0
                shape_str = "(no activation)"
            else:
                n_neurons = acts.shape[1] if acts.ndim >= 2 else 1
                shape_str = str(n_neurons)
            # module type
            module = dict(decoder.named_modules()).get(name)
            mtype = type(module).__name__ if module is not None else "?"
            print(f"{idx:>4}  {name:<46} {mtype:<18} {shape_str:>9}")
            structure.append((name, mtype, n_neurons))

        print("-" * 78)
        print(f"Total hooked layers: {len(structure)}")
        print(f"Channel range: {min(s[2] for s in structure if s[2]>0)} "
            f"– {max(s[2] for s in structure)}")
        print("=" * 78)

        # Reset so this dummy pass doesn't pollute a subsequent real run
        for hook in self.hooks.values():
            hook.reset()

        return structure

    def activate(self, audio_list, metadata_list, sample_rate=44100, min_duration_for_bpm=0.5):
        """
        Activate the analyser with audio and metadata.

        Args:
            audio_list: List of audio tensors
            metadata_list: List of metadata dicts
            sample_rate: Sample rate in Hz (default 44100)
            min_duration_for_bpm: Minimum audio duration (seconds) to keep BPM metadata
        """
        print("\n" + "="*60)
        print("Registering hooks on decoder layers")
        print("="*60)
        print(f"audio_list {len(audio_list)}")
        print(f"metadata_list {len(metadata_list)}")

        self.sample_rate = sample_rate

        # Remove BPM from samples too short to carry reliable rhythmic information
        cleaned_metadata_list = []
        bpm_removed_count = 0
        for audio, metadata in zip(audio_list, metadata_list):
            if audio.dim() == 1:
                duration = audio.shape[0] / sample_rate
            elif audio.dim() == 2:
                duration = audio.shape[1] / sample_rate if audio.shape[0] <= 2 else audio.shape[0] / sample_rate
            else:
                duration = audio.shape[-1] / sample_rate
            cleaned = metadata.copy()
            if duration < min_duration_for_bpm and 'bpm' in cleaned:
                del cleaned['bpm']
                bpm_removed_count += 1
            cleaned_metadata_list.append(cleaned)
        if bpm_removed_count > 0:
            print(f"  Removed BPM metadata from {bpm_removed_count} samples with duration < {min_duration_for_bpm}s")

        # First try standard layer types
        self.register_decoder_hooks()

        # If no hooks registered, try hooking all leaf modules
        if len(self.hooks) == 0:
            print("\n⚠️  No standard layers found, trying to hook ALL leaf modules...")
            print("This will hook custom layer types in your RAVE architecture.")
            self.register_decoder_hooks(hook_all_leaf_modules=True)

        if len(self.hooks) == 0:
            raise RuntimeError(
                "Could not register any hooks! Your RAVE model architecture may not be supported. "
                "Please check the decoder structure manually."
            )

        self.metadata_list = cleaned_metadata_list
        self._balanced_sample_cache = {}

        print("\n" + "="*60)
        print("Collecting activations")
        print("="*60)
        self.collect_activations(audio_list, batch_size=4)

    def set_balanced_feature_indices(self, feature_indices: Dict[str, set]):
        """
        Register precomputed per-feature balanced samples (see
        dataset/make_balanced_dataset.py + get_correlations_clusters.merge_balanced_dataset).

        feature_indices maps property_key -> set of positions in
        self.metadata_list that came from that feature's own balanced subset
        (each subset already carries the audio appropriate for that feature —
        short pitch segment for 'pitch', full chunk_duration window for
        'bpm'/'spectral_centroid'/'spectral_bandwidth'). Since
        merge_balanced_dataset() no longer deduplicates across features, the
        same underlying sample may appear at multiple positions (once per
        feature that selected it) — that's expected. When set,
        get_balanced_sample() uses this precomputed selection instead of
        recomputing balanced_sample live, guaranteeing it matches
        dataset_stats.py exactly.
        """
        self.balanced_feature_indices = feature_indices
        self._balanced_sample_cache = {}

    def get_balanced_sample(self, property_key: str, n: int = 500):
        """
        Return (valid_indices, property_values) for `property_key`, balanced
        across its value range and capped at `n` samples.

        If set_balanced_feature_indices() has been called, the precomputed
        per-feature positions are used directly, so results are identical to
        dataset_stats.py's balanced pkl. Otherwise falls back to recomputing
        balanced_sample live with a fixed seed (BALANCED_SAMPLE_SEED), applying
        the same bpm-pass/pitch-pass exclusion.

        Computed once per (analyser instance, property_key) and cached.
        """
        if property_key in self._balanced_sample_cache:
            return self._balanced_sample_cache[property_key]

        balanced_feature_indices = getattr(self, "balanced_feature_indices", None)

        if balanced_feature_indices is not None:
            # pitch_class isn't balanced separately — it rides along on the pitch pass.
            lookup_key = "pitch" if property_key == "pitch_class" else property_key
            wanted_positions = balanced_feature_indices.get(lookup_key, set())
            valid_indices, property_values = [], []
            for i in sorted(wanted_positions):
                meta = self.metadata_list[i]
                if property_key in meta and meta[property_key] is not None:
                    try:
                        property_values.append(float(meta[property_key]))
                        valid_indices.append(i)
                    except (ValueError, TypeError):
                        continue
            valid_indices = np.array(valid_indices)
            property_values = np.array(property_values, dtype=np.float32)

        else:
            # bpm lives only on BPM-pass entries; pitch/pitch_class on pitch-pass entries.
            # Excluding the other pass avoids BPM entries inflating pitch values when pyin
            # incidentally detects pitch on rhythmic material (mirrors dataset_stats.py).
            # NOTE: indices below stay relative to self.metadata_list (not the filtered
            # pool) since they're later used to index activation_records arrays.
            if property_key == "bpm":
                pool_indices = [i for i, m in enumerate(self.metadata_list) if "bpm" in m]
            elif property_key in ("pitch", "pitch_class"):
                pool_indices = [i for i, m in enumerate(self.metadata_list) if "bpm" not in m]
            else:
                pool_indices = range(len(self.metadata_list))

            valid_indices, property_values = [], []
            for i in pool_indices:
                meta = self.metadata_list[i]
                if property_key in meta and meta[property_key] is not None:
                    try:
                        property_values.append(float(meta[property_key]))
                        valid_indices.append(i)
                    except (ValueError, TypeError):
                        continue

            valid_indices = np.array(valid_indices)
            property_values = np.array(property_values, dtype=np.float32)

            if len(property_values) >= 2:
                rng = np.random.default_rng(BALANCED_SAMPLE_SEED)
                balanced = get_balanced_indices(property_values, n=n, feature_name=property_key, rng=rng)
                valid_indices = valid_indices[balanced]
                property_values = property_values[balanced]

        self._balanced_sample_cache[property_key] = (valid_indices, property_values)
        return valid_indices, property_values

    def register_decoder_hooks(self, layer_pattern: Optional[str] = None,
                              hook_all_leaf_modules: bool = False):
        """
        Register hooks on decoder layers to capture activations
        
        Args:
            layer_pattern: If provided, only register hooks on layers matching this pattern
                          If None, register on all decoder layers
            hook_all_leaf_modules: If True, hook ALL leaf modules regardless of type
                                   Useful for custom RAVE architectures
        """
        # Clear existing hooks
        self.remove_hooks()
        
        # Find decoder module (adjust based on RAVE architecture)
        decoder = self._find_decoder()
        
        candidates = 0
        registered = 0
        
        for name, module in decoder.named_modules():
            # Skip non-leaf modules and uninteresting layers
            if len(list(module.children())) > 0:
                continue
            
            candidates += 1
                
            # Filter by pattern if provided
            if layer_pattern and layer_pattern not in name:
                continue
            
            # Check if we should hook this module
            should_hook = False
            module_type = type(module).__name__
            
            if hook_all_leaf_modules:
                # Hook everything except activations and norms if requested
                excluded_types = ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'GELU', 'SiLU',
                                'BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'GroupNorm',
                                'Dropout', 'Dropout2d', 'Identity', 'Snake']
                should_hook = module_type not in excluded_types
            else:
                # Standard check for common layer types
                # For checkpoint models, we can check actual types
                should_hook = isinstance(module, (
                    nn.Conv1d, nn.ConvTranspose1d, nn.Linear,
                    nn.Conv2d, nn.ConvTranspose2d
                ))
                
                # Also check for RAVE-specific layer types by name
                # (in case they're custom classes)
                if not should_hook and any(keyword in module_type.lower() 
                                          for keyword in ['conv', 'linear', 'cached']):
                    should_hook = True
            
            if should_hook:
                hook = ActivationHook(name)
                handle = module.register_forward_hook(hook)
                
                self.hooks[name] = hook
                self.hook_handles.append(handle)
                registered += 1
                # print(f"  ✓ Registered hook on: {name} ({type(module).__name__})")
        
        if registered == 0:
            print("WARNING: No hooks were registered! Check decoder architecture.")
            print("\nAvailable leaf modules in decoder (showing all 112):")
            module_types = {}
            for name, module in decoder.named_modules():
                if len(list(module.children())) == 0:
                    module_type = type(module).__name__
                    if module_type not in module_types:
                        module_types[module_type] = []
                    module_types[module_type].append(name)
            
            print("\nModule types found:")
            for module_type, names in module_types.items():
                print(f"  {module_type}: {len(names)} instances")
                # Show first few examples
                for name in names[:3]:
                    print(f"    - {name}")
                if len(names) > 3:
                    print(f"    ... and {len(names) - 3} more")
            
            print("\nNeed to add these module types to the hook registration!")
            print("Look for Conv, Transpose, Upsample, or similar layer types.")
        # else:
        #     # Verify hooks are actually stored
        #     print(f"Verifying: self.hooks contains {len(self.hooks)} entries")
        #     print(f"Verifying: self.hook_handles contains {len(self.hook_handles)} entries")
                
    def _find_decoder(self) -> nn.Module:
        """
        Find the decoder module in RAVE
        This may need adjustment based on RAVE version
        """
        # Try common decoder attribute names
        for attr in ['decoder', 'decode', 'generator', 'gen', 'dec', 'synthesis']:
            if hasattr(self.model, attr):
                decoder = getattr(self.model, attr)
                # Make sure it's a module, not a method
                if isinstance(decoder, nn.Module):
                    print(f"Found decoder module: '{attr}' ({type(decoder).__name__})")
                    return decoder
        
        # If not found, return the whole model
        print("Warning: Could not find decoder module, using full model")
        print("This may hook encoder layers too, which is fine for exploration")
        return self.model
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()
        
    @torch.no_grad()
    def collect_activations(
        self,
        audio_inputs: List[torch.Tensor],
        batch_size: int = 16
    ):
        """
        Collect activations for a list of audio inputs
        
        Args:
            audio_inputs: List of audio tensors
            batch_size: Batch size for processing
        """
        latent_dim = 128
        # print(f"\nCollecting activations for {len(audio_inputs)} audio samples...")
        # print(f"Number of hooks registered: {len(self.hooks)}")
        
        # CRITICAL CHECK: Verify hooks still exist
        if len(self.hooks) == 0:
            print("ERROR: No hooks registered! Did you call register_decoder_hooks()?")
            print("Current state:")
            print(f"  - self.hooks: {self.hooks}")
            print(f"  - self.hook_handles: {len(self.hook_handles)} handles")
            raise RuntimeError("No hooks registered. Cannot collect activations.")
            
        # Reset all hooks
        for hook in self.hooks.values():
            hook.reset()

        # Group audio by length to avoid trimming within batches
        from collections import defaultdict
        length_groups = defaultdict(list)

        for idx, audio in enumerate(audio_inputs):
            # Get audio length (handling different tensor shapes)
            if audio.dim() == 1:
                length = audio.shape[0]
            elif audio.dim() == 2:
                length = audio.shape[1] if audio.shape[0] <= 2 else audio.shape[0]
            else:
                length = audio.shape[-1]

            length_groups[length].append(idx)

        # # Report length distribution
        # if len(length_groups) > 1:
        #     print(f"  Found {len(length_groups)} different audio lengths:")
        #     for length, indices in sorted(length_groups.items()):
        #         print(f"    - {length} samples: {len(indices)} files")

        # Process each length group separately
        total_batches = sum((len(indices) + batch_size - 1) // batch_size
                           for indices in length_groups.values())
        batch_counter = 0

        # Length-group batching processes samples out of original order (all
        # samples of one length, then all of the next, etc). Track the
        # original index of every sample in the order it's actually run
        # through the model, so activations can be permuted back afterwards
        # to align with audio_inputs/metadata_list order.
        processing_order = []

        for length in sorted(length_groups.keys()):
            indices = length_groups[length]
            n_batches_this_length = (len(indices) + batch_size - 1) // batch_size

            for batch_idx, i in enumerate(range(0, len(indices), batch_size)):
                batch_indices = indices[i:i+batch_size]
                batch = [audio_inputs[idx] for idx in batch_indices]
                processing_order.extend(batch_indices)
                batch_counter += 1

                # print(f"  Batch {batch_counter}/{total_batches} (length={length:.2f}s, {len(batch)} samples)", flush=True)

                # Ensure each audio has correct shape (add channel dimension if needed)
                processed_batch = []
                for audio in batch:
                    if audio.dim() == 1:
                        # Add channel dimension: (time,) -> (1, time)
                        audio = audio.unsqueeze(0)
                    elif audio.dim() == 2 and audio.shape[0] > audio.shape[1]:
                        # If shape is (time, channels), transpose to (channels, time)
                        audio = audio.transpose(0, 1)
                    processed_batch.append(audio)

                # Stack batch: (batch, channels, time)
                # No trimming needed since all audio in this batch has same length
                batch_tensor = torch.stack(processed_batch).to(self.device)

                # Forward pass through model
                if hasattr(self.model, 'decode'):
                    # If using encoder-decoder, first encode
                    z = self.model.encode(batch_tensor)
                    z = z[:, :latent_dim, :]
                    output = self.model.decode(z)
                else:
                    output = self.model(batch_tensor)

            # Activations are now stored in hooks
        
        # print("\n" + "-"*60)
        # print("DEBUG: Checking hook state after forward passes")
        # print(f"  Number of hooks in self.hooks: {len(self.hooks)}")
        # print(f"  Hook names: {list(self.hooks.keys())[:3]}...")
        # for name, hook in list(self.hooks.items())[:2]:
        #     print(f"  Hook '{name}' has {len(hook.activations)} activation batches")
        # print("-"*60)
        
        print("\nConverting activations to records...")
        # Undo the length-group reordering: restore_order[k] is the position in
        # the concatenated (processing-order) activations array of the sample
        # whose original index is k.
        restore_order = np.argsort(processing_order)

        # Convert activations to ActivationRecords
        for layer_name, hook in self.hooks.items():
            activations = hook.get_activations()
            if len(activations) == 0:
                print(f"  WARNING: No activations captured for layer {layer_name}")
                continue
            activations = activations[restore_order]
            self.activation_records[layer_name] = ActivationRecord(
                layer_name=layer_name,
                activations=activations
            )
            # print(f"  Layer {layer_name}: collected {activations.shape[0]} samples "
            #       f"with {activations.shape[1]} neurons")
    

    def match_names(self):
        def _get_layer(decoder, layer_name: str) -> nn.Module:
            """Get layer module by name"""
            for name, module in decoder.named_modules():
                if name == layer_name:
                    return module  
                
        for layer_name in self.cluster_results.keys():
            module = _get_layer(self.model.decoder, layer_name)
            
            if isinstance(module, nn.Conv1d):
                print(f"{layer_name}:")
                print(f"  Conv1d: {module.in_channels} -> {module.out_channels}")
                print(f"  Kernel: {module.kernel_size}, Stride: {module.stride}")
            elif isinstance(module, nn.ConvTranspose1d):
                print(f"{layer_name}:")
                print(f"  ConvTranspose1d: {module.in_channels} -> {module.out_channels}")
                print(f"  Kernel: {module.kernel_size}, Stride: {module.stride}")

    def compute_neuron_variance_correlation(
        self,
        layer_name: str,
        property_key: str = 'pitch'
    ) -> Dict:
        """
        Compute correlation between neuron variance and stimulus property variation

        This addresses question: "Would it be that particular neurons would have
        high variation as a particular property is varied?"

        Args:
            analyser: analyser with activations
            layer_name: Layer to analyze
            property_key: Property to correlate with (e.g., 'pitch', 'bpm', 'pitch_class')

        Returns:
            Dictionary with correlation results
        """
        # print(f"\nComputing variance-property correlation for {layer_name}...")

        record = self.activation_records[layer_name]
        activations = record.activations

        valid_indices, properties = self.get_balanced_sample(property_key, n=500)

        if len(properties) < 2:
            print(f"  Not enough samples with property '{property_key}'")
            return {}

        activations = activations[valid_indices]

        # Compute variance of each neuron across stimuli
        neuron_variances = np.var(activations, axis=0)

        # Vectorised Spearman: rank once, then a single matrix dot product
        # instead of calling spearmanr C times in a loop.
        from scipy.stats import rankdata
        from scipy.stats import t as _t_dist

        n = len(properties)
        prop_ranks = rankdata(properties).astype(np.float64)
        act_ranks  = np.apply_along_axis(rankdata, 0, activations.astype(np.float64))  # [N, C]

        prop_c = prop_ranks - prop_ranks.mean()
        act_c  = act_ranks  - act_ranks.mean(axis=0)

        num   = prop_c @ act_c                                              # [C]
        denom = np.sqrt((prop_c ** 2).sum() * (act_c ** 2).sum(axis=0))
        rho   = num / np.where(denom > 0, denom, 1e-10)
        rho   = np.clip(rho, -1 + 1e-10, 1 - 1e-10)

        t_stat  = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        p_values    = 2 * _t_dist.sf(np.abs(t_stat), df=n - 2)
        correlations = np.abs(rho)
        
        # Find neurons with high variance and high correlation
        high_var_threshold = np.percentile(neuron_variances, 75)
        high_corr_threshold = 0.5
        significant_alpha = 0.05
        
        responsive_neurons = np.where(
            (neuron_variances > high_var_threshold) &
            (np.abs(correlations) > high_corr_threshold) &
            (p_values < significant_alpha)
        )[0]
        
        num_responsive = len(responsive_neurons)
        responsive_neuron_std_correlation = 0
        responsive_neuron_mean_correlation = 0
        responsive_neuron_proportion = 0
        proportion = 0
        if num_responsive > 0:
            responsive_neuron_proportion = num_responsive/len(correlations)
            responsive_neuron_correlations = correlations[responsive_neurons]
            responsive_neuron_std_correlation = np.std(np.abs(responsive_neuron_correlations))
            responsive_neuron_mean_correlation = np.mean(np.abs(responsive_neuron_correlations))

        results = {
            'property': property_key,
            'n_responsive_neurons': num_responsive,
            "responsive_neuron_proportion":responsive_neuron_proportion,
            'responsive_neuron_indices': responsive_neurons.tolist(),
            'responsive_neuron_std_correlation': responsive_neuron_std_correlation,
            'responsive_neuron_mean_correlation': responsive_neuron_mean_correlation,
            'mean_correlation': np.mean(np.abs(correlations)),
            'max_correlation': np.max(np.abs(correlations)),
            'all_correlations': correlations.tolist()
        }
        
        return results  

    def load_correlation(self, output_dir):
        variance_path = Path(output_dir) / "variance_correlation.json"
        with open(variance_path, 'r') as f:
            from_file = json.load(f)
        self.correlations = from_file

    def do_correlation(self, output_dir, prop:List = ['pitch', 'pitch_class', 'bpm'], update: bool = False):
        """
        update=False (default): variance_correlation.json is fully overwritten
        with exactly `prop`'s results — any properties previously saved but not
        in `prop` are dropped.
        update=True: `prop` is recomputed and merged into the existing
        variance_correlation.json (per layer), leaving properties not in
        `prop` untouched. Use this to refresh only specific properties (e.g.
        just spectral_centroid/spectral_bandwidth after a dataset fix) without
        rerunning pitch/bpm.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print("\n" + "="*60)
        print("Additional Analysis: Neuron Variance-Property Correlation")
        print("="*60)

        layer_names = list(self.activation_records.keys())
        pairs = [(ln, p) for ln in layer_names for p in prop]

        # Threading is appropriate here — numpy releases the GIL so all cores
        # are used without the overhead of spawning separate processes.
        results_flat = {}
        with ThreadPoolExecutor() as ex:
            futures = {
                ex.submit(self.compute_neuron_variance_correlation, ln, p): (ln, p)
                for ln, p in pairs
            }
            for fut in as_completed(futures):
                ln, p = futures[fut]
                result = fut.result()
                if result:
                    results_flat[(ln, p)] = result

        variance_results = {}
        for ln in layer_names:
            layer_results = {p: results_flat[(ln, p)] for p in prop if (ln, p) in results_flat}
            if layer_results:
                variance_results[ln] = layer_results

        variance_path = Path(output_dir) / "variance_correlation.json"

        if update and variance_path.exists():
            with open(variance_path) as f:
                existing = json.load(f)
            for ln, layer_results in variance_results.items():
                existing.setdefault(ln, {}).update(layer_results)
            variance_results = existing

        with open(variance_path, 'w') as f:
            json.dump(variance_results, f, indent=2)
        print(f"\nSaved variance analysis to {variance_path}")

    def sort_correlations(self, prop:List = ['pitch', 'pitch_class', 'bpm']):
        res = {p:[] for p in prop}
        for k,v in self.correlations.items():
            for p in prop:
                try:
                    responsive = v[p]["n_responsive_neurons"]
                    total = len(v[p]["all_correlations"])
                    percent = (responsive/total)*100
                    cor = v[p]["mean_correlation"]
                    res[p].append([k, responsive, total, percent, cor])
                except KeyError as e:
                    print(f"no property {p}")
        
        for p in prop:
            res[p].sort(key = lambda x:x[3])
            print(f"{p} sorted proportion")
            for i in res[p]:
                print(i)
        
            res[p].sort(key = lambda x:x[4])
            print(f"{p} sorted correlation")
            for i in res[p]:
                print(i)

    def do_clustering(self, output_dir, n_clusters, pca_components):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        
        # 6. For each layer, perform clustering analysis
        print("\n" + "="*60)
        print("Step 5: Clustering analysis")
        print("="*60)
        
        results_summary = {}
        layers = list(self.activation_records.keys())
        # layers.append("all")
        for layer_name in layers:
            # print(f"\n--- Analyzing layer: {layer_name} ---")
            
            # Perform clustering
            cluster_result = self.cluster_neurons(
                layer_name,
                method='kmeans',
                n_clusters=n_clusters,
                use_pca=True,
                pca_components=pca_components
            )
            
            # Analyze what each cluster responds to
            cluster_properties = {}
            for meta_key in ['pitch', 'pitch_class', 'bpm']:
                try:
                    print(meta_key)
                    props = self.analyze_cluster_properties(layer_name, meta_key)
                    cluster_properties[meta_key] = props
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
            
            # Save cluster properties
            results_summary[layer_name] = {
                'silhouette_score':float(cluster_result.silhouette_score),
                'n_clusters': cluster_result.n_clusters,
                'cluster_labels': cluster_result.cluster_labels.tolist(),
                'cluster_properties': cluster_properties,
                'explained_variance': cluster_result.explained_variance.tolist() if cluster_result.explained_variance is not None else None
            }
            

        # 7. Save results summary
        results_path = output_dir / "clustering_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSaved results to {results_path}")

    def _assign_section(self, layer_name: str):
        """
        Map a hooked layer name to an early/middle/late section.

        Supports two naming schemes:
          RAVE     : 'net.X.something'   → split by net index (early<=7, middle 8-14, late 15+)
          EnCodec  : 'layers.X.conv...'  → split by upsampling stage (Option 2)

        The EnCodec decoder LSTM ('layers.1.lstm') is excluded from the
        analysis: it is the sole recurrent layer and is not part of the
        convolutional depth structure. Returns None for excluded layers.
        """
        # ---- EnCodec LSTM: exclude ----
        if 'lstm' in layer_name.lower():
            return None

        # ---- RAVE naming: net.X ----
        if layer_name.startswith('net.'):
            try:
                idx = int(layer_name.split('.')[1])
            except (IndexError, ValueError):
                return None
            if idx <= 7:
                return 'early'
            elif idx <= 14:
                return 'middle'
            else:
                return 'late'

        # ---- EnCodec naming: layers.X.conv ----
        if layer_name.startswith('layers.'):
            try:
                idx = int(layer_name.split('.')[1])
            except (IndexError, ValueError):
                return None
            # Option 2: group each upsampling stage with its following resblock.
            #   early  : input conv + stage 1 + resblock 1   → layers 0, 3, 4
            #   middle : stage 2 + resblock 2 + stage 3 + resblock 3 → layers 6, 7, 9, 10
            #   late   : stage 4 + resblock 4 + output conv → layers 12, 13, 15
            # (layer 1 = LSTM, already excluded above)
            if idx <= 4:
                return 'early'
            elif idx <= 10:
                return 'middle'
            else:
                return 'late'

        # Unknown naming scheme
        return None
    
    def do_cross_layer_clustering(self, output_dir, n_clusters, pca_components, variance_threshold=0.95):
        """
        Cluster neurons across layer sections (early, middle, late) instead of within each layer.
        Only includes highly responsive neurons based on variance threshold.
        Args:
            output_dir: Directory to save results
            n_clusters: Number of clusters
            pca_components: Number of PCA components
            variance_threshold: Keep top neurons accounting for this much variance (0-1)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "="*60)
        print("Cross-Layer Clustering Analysis")
        print("="*60)

        # Define layer sections
        layer_sections = {
            'early':  [],
            'middle': [],
            'late':   [],
        }

        for layer_name in self.activation_records.keys():
            section = self._assign_section(layer_name)
            if section is not None:
                layer_sections[section].append(layer_name)

        # print(f"\nLayer sections:")
        # for section, layers in layer_sections.items():
        #     print(f"  {section}: {len(layers)} layers")

        results_summary = {}

        # Process each section
        for section_name, layer_names in layer_sections.items():
            if len(layer_names) == 0:
                print(f"\n⚠️  No layers in {section_name} section, skipping...")
                continue

            # print(f"\n{'='*60}")
            # print(f"Processing {section_name} section ({len(layer_names)} layers)")
            # print(f"{'='*60}")

            # Collect activations from all layers in this section
            all_activations = []
            neuron_layer_map = []  # Track which layer and index each neuron belongs to

            for layer_name in layer_names:
                activations = self.activation_records[layer_name].activations.T
                neuron_variances = np.var(activations, axis=1)
                sorted_indices = np.argsort(neuron_variances)[::-1]
                #keep all
                top_indices = sorted_indices
                all_activations.append(activations[top_indices])
                # Store both layer name and original neuron index
                neuron_layer_map.extend([(layer_name, int(idx)) for idx in top_indices])

            # Combine all activations
            combined_activations = np.vstack(all_activations)
            # print(f"\n  Total neurons in {section_name}: {combined_activations.shape[0]}")
            # print(f"  Total samples: {combined_activations.shape[1]}")

            # Create temporary activation record for combined data to reuse existing functions
            temp_layer_name = f'section_{section_name}'
            temp_record = ActivationRecord(
                layer_name=temp_layer_name,
                activations=combined_activations.T
            )
            self.activation_records[temp_layer_name] = temp_record

            # Use existing cluster_neurons function which handles PCA and clustering
            # print(f"\n  Applying PCA and clustering...")
            cluster_result = self.cluster_neurons(
                layer_name=temp_layer_name,
                method='kmeans',
                n_clusters=n_clusters,
                use_pca=True,
                pca_components=pca_components
            )

            # Store cluster results for use with analyze_cluster_properties
            self.cluster_results[temp_layer_name] = cluster_result

            # print(f"  Silhouette score: {cluster_result.silhouette_score:.3f}")

            # Analyze cluster properties using existing function
            cluster_properties = {}
            for meta_key in ['pitch', 'pitch_class', 'bpm']:
                prop_analysis = self.analyze_cluster_properties(
                    layer_name=temp_layer_name,
                    metadata_key=meta_key,
                    top_k=20
                )
                if prop_analysis:
                    cluster_properties[meta_key] = prop_analysis
                    # print(f"    Analyzed {meta_key} properties")

            # Save results
            results_summary[section_name] = {
                'silhouette_score': float(cluster_result.silhouette_score),
                'n_clusters': cluster_result.n_clusters,
                'n_neurons': combined_activations.shape[0],
                'layers': layer_names,
                'neuron_layer_map': [[layer, idx] for layer, idx in neuron_layer_map],  # Convert tuples to lists for JSON
                'cluster_labels': cluster_result.cluster_labels.tolist(),
                'cluster_properties': cluster_properties,
                'explained_variance': cluster_result.explained_variance.tolist() if cluster_result.explained_variance is not None else []
            }

            # Note: cluster_result is already stored in self.cluster_results[temp_layer_name]

        # Save results
        results_path = output_dir / "cross_layer_clustering_results_all_neurons.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n✓ Saved cross-layer clustering results to {results_path}")

    def do_cross_layer_correlation(self, output_dir, prop: List = ['pitch', 'pitch_class', 'bpm'],
                                    update: bool = False):
        """
        Perform correlation analysis on cross-layer clusters.
        Reuses compute_neuron_variance_correlation by creating temporary activation records for each cluster.
        Reconstructs section activations from original layer records and neuron_layer_map.

        Args:
            output_dir: Directory to save results
            prop: List of properties to correlate with
            update: if False (default), cross_layer_cluster_correlation_all_neurons.json
                is fully overwritten with exactly `prop`'s results. If True, `prop` is
                recomputed and merged into the existing file's 'properties' per
                (section, cluster), leaving properties not in `prop` untouched — use
                this to refresh only specific properties without rerunning the rest.
        """
        print("\n" + "="*60)
        print("Cross-Layer Cluster Correlation Analysis")
        print("="*60)

        # Load cross-layer clustering results to get neuron_layer_map and cluster_labels
        output_dir = Path(output_dir)
        clustering_results_path = output_dir / "cross_layer_clustering_results_all_neurons.json"

        if not clustering_results_path.exists():
            raise FileNotFoundError(f"Cross-layer clustering results not found at {clustering_results_path}. "
                                    "Run do_cross_layer_clustering first.")

        with open(clustering_results_path, 'r') as f:
            clustering_results = json.load(f)

        print(f"Loaded clustering results from {clustering_results_path}")

        correlation_results = {}

        # Iterate over all sections
        for section_name in ['early', 'middle', 'late']:
            if section_name not in clustering_results:
                print(f"\n⚠️  No clustering results for {section_name} section, skipping...")
                continue

            # print(f"\n{'='*60}")
            # print(f"Analyzing {section_name} section clusters")
            # print(f"{'='*60}")

            section_data = clustering_results[section_name]
            neuron_layer_map = section_data['neuron_layer_map']
            cluster_labels = np.array(section_data['cluster_labels'])
            n_clusters = section_data['n_clusters']

            # Reconstruct section activations from original layer records using neuron_layer_map
            # print(f"  Reconstructing section activations from {len(neuron_layer_map)} neurons...")

            # Get activations for each neuron from its original layer
            all_neuron_activations = []
            for layer_name, neuron_idx in neuron_layer_map:
                if layer_name not in self.activation_records:
                    raise ValueError(f"Layer {layer_name} not found in activation_records. "
                                   "Ensure activations have been collected for all layers.")

                # Get activations for this specific neuron [samples, neurons] -> [samples]
                layer_activations = self.activation_records[layer_name].activations
                neuron_activation = layer_activations[:, neuron_idx]
                all_neuron_activations.append(neuron_activation)

            # Stack into [samples, neurons]
            section_activations = np.column_stack(all_neuron_activations)
            # print(f"  Reconstructed section activations shape: {section_activations.shape}")

            section_correlation_results = {}

            # Analyze each cluster
            for cluster_id in range(n_clusters):
                # print(f"\n  Cluster {cluster_id}:")

                # Get neurons belonging to this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_neuron_indices = np.where(cluster_mask)[0]
                n_neurons_in_cluster = len(cluster_neuron_indices)

                # print(f"    Neurons in cluster: {n_neurons_in_cluster}")

                # Get the specific neurons (layer, index) for this cluster
                cluster_neuron_origins = [neuron_layer_map[i] for i in cluster_neuron_indices]

                # Extract activations for only this cluster's neurons
                cluster_activations = section_activations[:, cluster_neuron_indices]

                # Create temporary activation record for this cluster
                temp_cluster_name = f"section_{section_name}_cluster_{cluster_id}"
                temp_record = ActivationRecord(
                    layer_name=temp_cluster_name,
                    activations=cluster_activations
                )
                self.activation_records[temp_cluster_name] = temp_record

                # Use existing compute_neuron_variance_correlation for each property
                cluster_prop_results = {}
                for p in prop:
                    result = self.compute_neuron_variance_correlation(
                        temp_cluster_name, property_key=p
                    )
                    if result:
                        cluster_prop_results[p] = result
                        # print(f"      {p}: mean_corr={result['mean_correlation']:.3f}, "
                        #       f"n_responsive={result['n_responsive_neurons']}")

                # Store results with neuron origins
                section_correlation_results[f"cluster_{cluster_id}"] = {
                    'properties': cluster_prop_results,
                    'neuron_origins': cluster_neuron_origins,
                    'n_neurons': n_neurons_in_cluster
                }

                # Clean up temporary record
                del self.activation_records[temp_cluster_name]

            correlation_results[section_name] = section_correlation_results

        # Save correlation results
        results_path = output_dir / "cross_layer_cluster_correlation_all_neurons.json"

        if update and results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)
            for section_name, section_results in correlation_results.items():
                existing_section = existing.setdefault(section_name, {})
                for cluster_key, cluster_result in section_results.items():
                    existing_cluster = existing_section.setdefault(
                        cluster_key, {'properties': {}, 'neuron_origins': [], 'n_neurons': 0})
                    existing_cluster['properties'].update(cluster_result['properties'])
                    existing_cluster['neuron_origins'] = cluster_result['neuron_origins']
                    existing_cluster['n_neurons'] = cluster_result['n_neurons']
            correlation_results = existing

        with open(results_path, 'w') as f:
            json.dump(correlation_results, f, indent=2)

        print(f"\n✓ Saved cross-layer cluster correlation results to {results_path}")

        return correlation_results

    def do_permutation_baseline(
        self,
        output_dir,
        prop: List = ['pitch', 'bpm'],
        n_permutations: int = 1000,
        threshold: float = 0.15,
        update: bool = False,
        force: bool = False,
    ) -> Dict:
        """
        Permutation baseline: shuffle feature labels and recompute layer correlations.

        For each permutation, property values are randomly reassigned across audio samples
        while activations are held fixed. This establishes the null distribution of |r|
        and the false-positive rate at a given threshold.

        Args:
            output_dir:     Directory to save results JSON.
            prop:           Feature keys to test (must exist in metadata_list).
            n_permutations: Number of shuffle iterations.
            threshold:      |r| threshold to evaluate false-positive rate against.
            update:         merge into the existing permutation_baseline.json rather
                             than overwriting it outright.
            force:          with update=True, normally properties already present in
                             the existing file are skipped (resume semantics). Set
                             force=True to always recompute everything in `prop` and
                             overwrite those entries, while properties NOT in `prop`
                             are still left untouched — use this to refresh specific
                             properties (e.g. after a dataset fix) without rerunning
                             everything else.

        Saves:
            permutation_baseline.json  with per-property:
              - null_mean, null_std       (mean/std of null |r| distribution)
              - pct_exceeding_threshold   (% of null correlations > threshold)
              - p95_threshold             (95th-percentile of null distribution)
              - observed_mean, observed_max  (real data, for comparison)
        """
        out_path = Path(output_dir) / "permutation_baseline.json"
        if update and out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            if not force:
                prop = [p for p in prop if p not in existing]
                if not prop:
                    print("  All features already present, nothing to compute.")
                    return existing
        else:
            existing = {}

        print("\n" + "="*60)
        print("Permutation Baseline Analysis")
        print("="*60)
        print(f"  n_permutations={n_permutations}, threshold=|r|>{threshold}")

        output_dir = Path(output_dir)
        results = {}

        for property_key in prop:
            print(f"\n--- {property_key} ---")

            valid_indices, property_values = self.get_balanced_sample(property_key, n=500)

            if len(property_values) < 2:
                print(f"  Not enough samples, skipping.")
                continue

            n_samples = len(property_values)
            print(f"  Samples used: {n_samples}  (balanced sample)")

            # Collect all per-neuron |r| from every layer (observed).
            # section_* layers are composite/derived records and are excluded.
            observed_corrs = []
            layer_activation_cache = {}
            for layer_name, record in self.activation_records.items():
                if layer_name.startswith("section_"):
                    continue
                acts = record.activations[valid_indices]   # [samples, neurons]
                layer_activation_cache[layer_name] = acts
                for neuron_idx in range(acts.shape[1]):
                    r, _ = spearmanr(property_values, acts[:, neuron_idx])
                    observed_corrs.append(abs(r))

            observed_corrs = np.array(observed_corrs)
            n_total = len(observed_corrs)
            print(f"  Total neuron×layer pairs: {n_total}")
            print(f"  Observed  mean|r|={observed_corrs.mean():.4f}  "
                  f"max|r|={observed_corrs.max():.4f}  "
                  f"%>{threshold}={100*(observed_corrs>threshold).mean():.1f}%")
            
            # Permutation loop — parallelised across CPU cores
            rng = np.random.default_rng(seed=42)
            shuffled_arrays = [rng.permutation(property_values) for _ in range(n_permutations)]

            print(f"    Running {n_permutations} permutations in parallel...", flush=True)
            null_all_corrs = Parallel(n_jobs=-1, backend='loky', verbose=1)(
                delayed(_run_one_permutation)(s, layer_activation_cache, threshold)
                for s in shuffled_arrays
            )
            null_all_corrs  = [np.array(c) for c in null_all_corrs]
            null_mean_r     = np.array([c.mean()                   for c in null_all_corrs])
            null_exceeding  = np.array([(c > threshold).mean()*100 for c in null_all_corrs])
            # p95 of pooled individual null |r| — used to threshold individual observed neurons
            null_pooled = np.concatenate(null_all_corrs)
            null_p95_individual = float(np.percentile(null_pooled, 95))
            obs_pct_exceeding_p95 = float((observed_corrs > null_p95_individual).mean() * 100)

            print(f"\n  Null mean|r|:            {null_mean_r.mean():.4f} ± {null_mean_r.std():.4f}")
            print(f"  Null p95 (individual):   {null_p95_individual:.4f}")
            print(f"  Obs %>null_p95:          {obs_pct_exceeding_p95:.1f}%")
            print(f"  Null %>|r|>{threshold}: {null_exceeding.mean():.2f}% ± {null_exceeding.std():.2f}%")
            print("...", flush=True)
            results[property_key] = {
                'n_permutations':              n_permutations,
                'n_neuron_layer_pairs':        int(n_total),
                'threshold':                   threshold,
                'null_mean_r':                 float(null_mean_r.mean()),
                'null_std_r':                  float(null_mean_r.std()),
                'null_p95_r':                  null_p95_individual,
                'null_pct_exceeding':          float(null_exceeding.mean()),
                'null_pct_exceeding_std':      float(null_exceeding.std()),
                'observed_mean_r':             float(observed_corrs.mean()),
                'observed_max_r':              float(observed_corrs.max()),
                'observed_pct_exceeding':      float((observed_corrs > threshold).mean() * 100),
                'observed_pct_exceeding_p95':  obs_pct_exceeding_p95,
            }

        results = {**existing, **results}
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved permutation baseline to {out_path}")
        
        return results

    def print_obs_r(self, prop: List = ['pitch', 'bpm', 'spectral_centroid', 'spectral_bandwidth']):
        """Print mean observed |r| per feature using the same sample and layer filtering as do_permutation_baseline."""
        for property_key in prop:
            valid_indices, property_values = self.get_balanced_sample(property_key, n=500)

            if len(property_values) < 2:
                print(f"{property_key}: not enough samples")
                continue

            observed_corrs = []
            for layer_name, record in self.activation_records.items():
                if layer_name.startswith("section_"):
                    continue
                acts = record.activations[valid_indices]
                for neuron_idx in range(acts.shape[1]):
                    r, _ = spearmanr(property_values, acts[:, neuron_idx])
                    observed_corrs.append(abs(r))

            observed_corrs = np.array(observed_corrs)
            print(f"{property_key}: mean|r|={observed_corrs.mean():.4f}  "
                  f"max|r|={observed_corrs.max():.4f}  "
                  f"n_neurons={len(observed_corrs)}  (balanced sample)")

    def sort_clusters(self):
        print("silhouette_score sorted")
        silhouette = [[k, v.silhouette_score] for k, v in self.cluster_results.items()]
        silhouette.sort(key = lambda x:x[1])
        for i in silhouette:
            print(i)
        return silhouette

   
    def perform_pca(
        self,
        layer_name: str,
        n_components: int = 2,
        normalize: bool = True
    ) -> Tuple[np.ndarray, PCA]:
        """
        Perform PCA on activations from a specific layer
        
        Args:
            layer_name: Name of layer to analyze
            n_components: Number of PCA components
            normalize: Whether to standardize before PCA
            
        Returns:
            Transformed activations, fitted PCA model
        """
        activations = self.get_activations(layer_name)
        print(activations.shape)
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            activations = scaler.fit_transform(activations)
        
        
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(activations)
        
        self.pca_models[layer_name] = pca
        
        return transformed, pca
    
    def get_activations(self, layer_name):
        # Get activations
        activations = np.zeros((1,36))
        if not layer_name == "all":
            activations = self.activation_records[layer_name].activations.T
        else:
            for l in self.activation_records.keys():
                activations = np.vstack((activations, self.activation_records[l].activations.T))
        return activations 

    def cluster_neurons(
        self,
        layer_name: str,
        method: str = 'kmeans',
        n_clusters: int = 8,
        use_pca: bool = True,
        pca_components: int = 10,
        **kwargs
    ) -> ClusterResult:
        """
        Cluster neurons based on their activation patterns
        
        Args:
            layer_name: Layer to cluster
            method: 'kmeans', 'dbscan', or other sklearn clustering methods
            n_clusters: Number of clusters (for kmeans)
            use_pca: Whether to use PCA before clustering
            pca_components: Number of PCA components if use_pca=True
            **kwargs: Additional arguments for clustering algorithm
            
        Returns:
            ClusterResult object
        """
        if layer_name not in self.activation_records and not layer_name=="all":
            raise ValueError(f"No activations for layer {layer_name}")
        
        activations = self.get_activations(layer_name)
        
        
        # Apply PCA if requested
        pca_components = min(pca_components,activations.shape[0],activations.shape[1])
        # print(f"pca_components {pca_components}")
        pca_comps = None
        explained_var = None
        if use_pca:
            activations, pca = self.perform_pca(
                layer_name, n_components=pca_components
            )
            pca_comps = pca.components_
            explained_var = pca.explained_variance_ratio_
        score = 0
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            labels = clusterer.fit_predict(activations)
            centers = clusterer.cluster_centers_
            score = silhouette_score(activations, labels)
        elif method == 'dbscan':
            clusterer = DBSCAN(**kwargs)
            labels = clusterer.fit_predict(activations)
            centers = None
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            mask = labels != -1
            try:
                score = silhouette_score(activations[mask], labels[mask])
            except Exception as e:
                print(e)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # print(f"silhouette score {score}")
        result = ClusterResult(
            silhouette_score=score,
            cluster_labels=labels,
            cluster_centers=centers,
            n_clusters=n_clusters,
            method=method,
            pca_components=pca_comps,
            explained_variance=explained_var
        )
        
        self.cluster_results[layer_name] = result
        # print(f"Clustered {layer_name} into {n_clusters} groups")
        
        return result
    
    def load_cluster_results(self, output_dir):
        # Check if we have any cluster results
        if not self.cluster_results:
            results_path = Path(output_dir) / "clustering_results.json"
            with open(results_path, 'r') as f:
                from_file = json.load(f)
                for layer_name in from_file.keys():
                    res_dict = from_file[layer_name]
                    result = ClusterResult(
                        silhouette_score=res_dict["silhouette_score"],
                        cluster_labels=res_dict["cluster_labels"],
                        n_clusters=res_dict["n_clusters"],
                        explained_variance=res_dict["explained_variance"]
                    )
                    self.cluster_results[layer_name] = result
        
    def analyze_cluster_properties(
        self,
        layer_name: str,
        metadata_key: str,
        top_k: int = 5
    ) -> Dict:
        """
        Analyze which stimuli most activate each neuron cluster

        Args:
            layer_name: Layer to analyze
            metadata_key: Property to examine (e.g., 'pitch_class', 'bpm')
            top_k: Number of top-activating samples to consider per cluster

        Returns:
            Dict mapping cluster_id -> Counter of property values
            Example: {0: {'pitch_class': {440: 3, 880: 2}}, 1: {'pitch_class': {110: 4, 220: 1}}}
        """
        #print(f"analyze_cluster_properties for {metadata_key}")
        activations = self.get_activations(layer_name)

        #list of labels givein clutser for each neuron
        labels = self.cluster_results[layer_name].cluster_labels
        metadata = self.metadata_list

        # Extract property values for each sample
        properties = [m.get(metadata_key) for m in metadata]

        # Ensure we only use as many samples as we have metadata for
        n_samples = len(properties)
        activations = activations[:, :n_samples]

        cluster_properties = {}

        for cluster_id in range(self.cluster_results[layer_name].n_clusters):
            # Get neurons belonging to this cluster
            cluster_neuron_indices = np.where(labels == cluster_id)[0]
            # Get activations for these neurons (get subset of neurons, all features)
            cluster_activations = activations[cluster_neuron_indices,:]
            # Average activation across cluster's neurons for each sample
            #so we are looking for the average activation of all neurons in this cluster
            sample_activation = cluster_activations.mean(axis=0)
            # Top K features indices (based on activation averaged across all neurons)
            top_features = np.argsort(sample_activation)[-top_k:]
            # Bounds check and filter None values
            cluster_props = [properties[i] for i in top_features if i < len(properties) and properties[i] is not None]
            # Count occurrences
            from collections import Counter
            counts = Counter(cluster_props)
            counts = {str(k): v for k, v in counts.items()}
            cluster_properties[f"cluster_{cluster_id}"] = counts

        # print(layer_name, metadata_key, cluster_properties)
        return cluster_properties
    
    def do_band_out_diagnostic_nonlinear(
        self,
        output_dir,
        prop: List = ['pitch', 'bpm'],
        n_bands: int = 5,
        n_permutations: int = 100,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        include_linear_baseline: bool = True,
        random_state: int = 42,
    ) -> Dict:
        """
        Band-out cross-validation diagnostic for nonlinear probes.

        Computes band-out R² (and its null distribution under shuffled labels)
        for each (layer, feature) pair. The companion function
        do_permutation_baseline_nonlinear computes the standard random-CV R²
        for the same cells — comparing the two reveals how much of the
        random-CV R² is attributable to dense local sampling vs. genuine
        feature-axis learning.

        Hidden dimensionality is scaled per layer based on input channel count
        (`min(64, max(8, n_channels // 8))`) to keep probe capacity proportional
        to input dimensionality. The same scaled value is used for the observed
        probe and its null distribution to ensure a valid comparison.

        Pitch and spectral centroid use log-spaced bands (matches musical/perceptual
        scale for frequency-domain quantities). BPM and spectral flatness use
        linear-spaced bands. Each band serves as a held-out test set in turn,
        with remaining bands as training. No band-to-band leakage is possible.

        Saves:
            band_out_diagnostic_nonlinear.json with per-feature, per-layer:
            - n_channels:            input channels at this layer
            - hidden_dim:            probe hidden dimensionality used (scaled per layer)
            - band_out_r2:           mean R² across n_bands held-out bands
            - band_out_r2_per_band:  per-band R² values (one per held-out band)
            - band_out_null_mean:    mean of null distribution under shuffled labels
            - band_out_null_p95:     95th percentile of null distribution
            - band_out_exceeds_p95:  bool
            - linear_band_out_r2:    (optional) linear-probe equivalent
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        print("\n" + "=" * 60)
        print("Band-Out CV Diagnostic for Nonlinear Probes")
        print("=" * 60)
        print(f"  n_bands={n_bands}, n_permutations={n_permutations}")
        print("  Hidden_dim scaled per layer as min(64, max(8, n_channels // 8))")
        print("  (Compare results against permutation_baseline_nonlinear.json "
            "for random-CV baseline)")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        results = {}

        def assign_bands(values: np.ndarray, feature: str, n_bands: int) -> np.ndarray:
            # Quantile-based edges: each band contains ~equal samples regardless of
            # distribution shape. Works for skewed features (spectral_bandwidth near 0),
            # log-normally-distributed features (spectral_centroid), and discrete-valued
            # features (bpm). np.unique collapses duplicate edges that arise when many
            # samples share the same value (e.g. integer BPM clusters).
            edges = np.unique(np.percentile(values, np.linspace(0, 100, n_bands + 1)))
            return np.digitize(values, edges[1:-1]).astype(int)

        for property_key in prop:
            print(f"\n--- {property_key} ---")

            valid_indices, property_values = self.get_balanced_sample(property_key, n=500)

            if len(property_values) < n_bands * 5:
                print(f"  Not enough samples ({len(property_values)}) for {n_bands} bands, "
                    f"skipping.")
                continue

            n_samples = len(property_values)

            bands = assign_bands(property_values, property_key, n_bands)
            unique_bands, band_counts = np.unique(bands, return_counts=True)
            print(f"  Valid samples: {n_samples}")
            print(f"  Band sizes: {dict(zip(unique_bands.tolist(), band_counts.tolist()))}")

            if len(unique_bands) < n_bands:
                print(f"  Warning: only {len(unique_bands)} bands populated, "
                    f"expected {n_bands}.")

            property_results = {}

            for layer_name, record in self.activation_records.items():
                if layer_name.startswith("section_") or "cluster_" in layer_name:
                    continue

                acts = record.activations[valid_indices].astype(np.float32)
                n_channels = acts.shape[1]
                layer_hidden_dim = min(64, max(8, n_channels // 8))

                # ---------- Observed: band-out CV with true labels ----------
                band_r2_folds = self._train_probe_band_out(
                    X=acts, y=property_values, bands=bands,
                    hidden_dim=layer_hidden_dim,
                    n_epochs=n_epochs, lr=learning_rate,
                    batch_size=batch_size,
                    early_stopping_patience=early_stopping_patience,
                    random_state=random_state,
                )
                band_out_r2 = float(np.mean(band_r2_folds))

                # ---------- Linear-probe band-out (optional) ----------
                linear_band_out_r2 = None
                if include_linear_baseline:
                    lin_band_r2_folds = self._train_probe_band_out(
                        X=acts, y=property_values, bands=bands,
                        hidden_dim=0,  # 0 -> linear probe (no hidden layer)
                        n_epochs=n_epochs, lr=learning_rate,
                        batch_size=batch_size,
                        early_stopping_patience=early_stopping_patience,
                        random_state=random_state,
                    )
                    linear_band_out_r2 = float(np.mean(lin_band_r2_folds))

                # ---------- Null: shuffled labels under band-out CV ----------
                # CRITICAL: use layer_hidden_dim here too — the null must use the
                # same probe capacity as the observed run for the comparison to be valid.
                rng = np.random.default_rng(seed=random_state)
                shuffled_arrays = [rng.permutation(property_values)
                                for _ in range(n_permutations)]

                print(f"  [{layer_name}] running {n_permutations} band-out permutations "
                    f"(n_channels={n_channels}, hidden_dim={layer_hidden_dim})...",
                    flush=True)

                # Dispatch each (permutation × band) as a separate task.
                # Old approach ran n_bands folds serially inside each of n_permutations
                # workers; this gives n_permutations × n_bands independent tasks for
                # better load-balancing across cores.
                n_unique_bands = len(unique_bands)
                all_null_flat = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                    delayed(_run_one_band_out_fold)(
                        acts, shuffled, bands, test_band,
                        layer_hidden_dim, n_epochs, learning_rate,
                        batch_size, early_stopping_patience,
                        random_state + i * n_unique_bands + band_idx,
                    )
                    for i, shuffled in enumerate(shuffled_arrays)
                    for band_idx, test_band in enumerate(unique_bands)
                )
                # reshape [n_permutations × n_bands] → [n_permutations, n_bands],
                # then mean per permutation to get one null R² per permutation
                null_r2 = np.nanmean(
                    np.array(all_null_flat, dtype=np.float32).reshape(
                        n_permutations, n_unique_bands
                    ),
                    axis=1,
                )
                null_mean = float(np.mean(null_r2))
                null_p95 = float(np.percentile(null_r2, 95))

                entry = {
                    'n_channels': int(n_channels),
                    'hidden_dim': int(layer_hidden_dim),
                    'n_samples': int(n_samples),
                    'n_bands': int(n_bands),
                    'band_out_r2': band_out_r2,
                    'band_out_r2_per_band': [float(x) for x in band_r2_folds],
                    'band_out_null_mean': null_mean,
                    'band_out_null_p95': null_p95,
                    'band_out_exceeds_p95': bool(band_out_r2 > null_p95),
                }
                if include_linear_baseline:
                    entry['linear_band_out_r2'] = linear_band_out_r2

                print(f"    band-out R² = {band_out_r2:.4f}, "
                    f"null p95 = {null_p95:.4f}"
                    + (f", linear band-out = {linear_band_out_r2:.4f}"
                        if include_linear_baseline else ""),
                    flush=True)

                property_results[layer_name] = entry

            results[property_key] = property_results

        out_path = output_dir / "band_out_diagnostic_nonlinear.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved band-out diagnostic to {out_path}")
        return results


    def _train_probe_band_out(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bands: np.ndarray,
        hidden_dim: int,
        n_epochs: int,
        lr: float,
        batch_size: int,
        early_stopping_patience: int,
        random_state: int,
    ) -> List[float]:
        """
        Train probes under band-out CV. Each fold holds out one band as test set.
        Returns list of held-out R² scores, one per band.
        """
        from sklearn.preprocessing import StandardScaler

        unique_bands = np.unique(bands)
        fold_args = []

        for fold_idx, test_band in enumerate(unique_bands):
            test_mask = bands == test_band
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(y_test) < 2 or len(y_train) < 2:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)

            y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
            y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
            y_test_n = ((y_test - y_mean) / y_std).astype(np.float32)
            fold_args.append((X_train, y_train_n, X_test, y_test_n, random_state + fold_idx))

        return Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(_fit_probe_once)(
                Xtr, ytr, Xte, yte,
                hidden_dim=hidden_dim, n_epochs=n_epochs, lr=lr,
                batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                seed=seed,
            )
            for Xtr, ytr, Xte, yte, seed in fold_args
        )
    
    def do_permutation_baseline_nonlinear(
        self,
        output_dir,
        prop: List = ['pitch', 'bpm'],
        n_permutations: int = 100,
        n_folds: int = 5,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        include_linear_baseline: bool = True,
        random_state: int = 42,
        update: bool = False,
        force: bool = False,
    ) -> Dict:
        """
        update: merge into the existing permutation_baseline_nonlinear.json
            rather than overwriting it outright.
        force: with update=True, normally properties already present in the
            existing file are skipped (resume semantics). Set force=True to
            always recompute everything in `prop` and overwrite those entries,
            while properties NOT in `prop` are still left untouched — use this
            to refresh specific properties without rerunning everything else.
        """

        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        out_path = output_dir / "permutation_baseline_nonlinear.json"
        if update and out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            if not force:
                prop = [p for p in prop if p not in existing]
                if not prop:
                    print("  All features already present, nothing to compute.")
                    return existing
        else:
            existing = {}

        print("\n" + "=" * 60)
        print("Nonlinear-Probe Permutation Baseline Analysis (random CV)")
        print("=" * 60)
        print(f"  n_permutations={n_permutations}, n_folds={n_folds}, n_epochs={n_epochs}")
        print("  Hidden_dim scaled per layer as min(64, max(8, n_channels // 8))")
        if include_linear_baseline:
            print(f"  Linear-probe baseline: enabled")

        results = {}

        for property_key in prop:
            print(f"\n--- {property_key} ---")

            # ---- Extract aligned (sample_idx, property_value) pairs ----
            valid_indices, property_values = self.get_balanced_sample(property_key, n=500)

            if len(property_values) < n_folds * 2:
                print(f"  Not enough samples ({len(property_values)}), skipping.")
                continue

            n_samples = len(property_values)
            print(f"  Valid samples: {n_samples}  (balanced sample)")

            property_results = {}

            # ---- Per-layer probing ----
            for layer_name, record in self.activation_records.items():
                # Skip the temporary cross-layer / cluster records that may be in
                # activation_records from earlier analyses
                if layer_name.startswith("section_") or "cluster_" in layer_name:
                    continue

                acts = record.activations[valid_indices].astype(np.float32)  # [N, C]
                n_channels = acts.shape[1]
                layer_hidden_dim = min(64, max(8, n_channels // 8))

                # ---------- Observed: nonlinear probe, true labels, CV ----------
                obs_r2_folds = self._train_probe_cv(
                    X=acts, y=property_values,
                    hidden_dim=layer_hidden_dim, n_folds=n_folds,
                    n_epochs=n_epochs, lr=learning_rate,
                    batch_size=batch_size,
                    early_stopping_patience=early_stopping_patience,
                    random_state=random_state,
                )
                observed_r2 = float(np.mean(obs_r2_folds))
                observed_r2_std = float(np.std(obs_r2_folds))

                # ---------- Observed: linear probe (optional) ----------
                linear_observed_r2 = None
                if include_linear_baseline:
                    lin_obs_r2_folds = self._train_probe_cv(
                        X=acts, y=property_values,
                        hidden_dim=0,  # 0 -> linear probe (no hidden layer)
                        n_folds=n_folds,
                        n_epochs=n_epochs, lr=learning_rate,
                        batch_size=batch_size,
                        early_stopping_patience=early_stopping_patience,
                        random_state=random_state,
                    )
                    linear_observed_r2 = float(np.mean(lin_obs_r2_folds))

                # ---------- Null: nonlinear probe, shuffled labels ----------
                # CRITICAL: use layer_hidden_dim here — the null must match the
                # observed probe's capacity for the comparison to be valid.
                # One R^2 per permutation (single train/test split inside each
                # permutation — the permutations themselves provide the noise estimate).
                rng = np.random.default_rng(seed=random_state)
                shuffled_arrays = [rng.permutation(property_values) for _ in range(n_permutations)]

                print(f"  [{layer_name}] running {n_permutations} permutations "
                    f"(n_channels={n_channels}, hidden_dim={layer_hidden_dim})...",
                    flush=True)

                # ---------- Null: nonlinear + linear in one Parallel call ----------
                nonlinear_null_jobs = [
                    delayed(_run_one_probe_permutation)(
                        acts, shuffled, layer_hidden_dim, n_epochs, learning_rate,
                        batch_size, early_stopping_patience, random_state + i,
                    )
                    for i, shuffled in enumerate(shuffled_arrays)
                ]
                linear_null_jobs = [
                    delayed(_run_one_probe_permutation)(
                        acts, shuffled, 0, n_epochs, learning_rate,
                        batch_size, early_stopping_patience, random_state + i,
                    )
                    for i, shuffled in enumerate(shuffled_arrays)
                ] if include_linear_baseline else []

                all_null = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                    nonlinear_null_jobs + linear_null_jobs
                )
                null_r2 = np.array(all_null[:n_permutations], dtype=np.float32)
                linear_null_mean_r2 = (
                    float(np.mean(all_null[n_permutations:])) if include_linear_baseline else None
                )

                null_mean = float(np.mean(null_r2))
                null_std = float(np.std(null_r2))
                null_p95 = float(np.percentile(null_r2, 95))

                ratio = (observed_r2 / null_mean) if abs(null_mean) > 1e-6 else float('nan')

                entry = {
                    'n_channels': int(n_channels),
                    'hidden_dim': int(layer_hidden_dim),
                    'n_samples': int(n_samples),
                    'null_mean_r2': null_mean,
                    'null_std_r2': null_std,
                    'null_p95_r2': null_p95,
                    'observed_r2': observed_r2,
                    'observed_r2_std': observed_r2_std,
                    'exceeds_null_p95': bool(observed_r2 > null_p95),
                    'delta_r2': observed_r2 - null_mean,
                    'ratio': ratio,
                }
                if include_linear_baseline:
                    entry['linear_observed_r2'] = linear_observed_r2
                    entry['linear_null_mean_r2'] = linear_null_mean_r2
                    if linear_observed_r2 is not None:
                        entry['nonlinear_gain'] = observed_r2 - linear_observed_r2

                print(f"    obs R^2 = {observed_r2:.4f} ± {observed_r2_std:.4f}, "
                    f"null p95 = {null_p95:.4f}, "
                    f"Δ = {entry['delta_r2']:+.4f}"
                    + (f", linear obs = {linear_observed_r2:.4f}, "
                        f"nonlin gain = {entry['nonlinear_gain']:+.4f}"
                        if include_linear_baseline else ""),
                    flush=True)

                property_results[layer_name] = entry

            results[property_key] = property_results

        results = {**existing, **results}
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved nonlinear-probe permutation baseline to {out_path}")
        return results


    def do_permutation_baseline_nonlinear_clusters(
        self,
        output_dir,
        prop: List = ['pitch', 'bpm'],
        n_permutations: int = 100,
        n_folds: int = 5,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        include_linear_baseline: bool = True,
        random_state: int = 42,
        update: bool = False,
        force: bool = False,
    ) -> Dict:
        """
        Same as do_permutation_baseline_nonlinear but probes each cross-layer
        cluster (neuron subset) rather than whole layers.

        Requires cross_layer_clustering_results_all_neurons.json in output_dir
        (produced by do_cross_layer_clustering).

        update=True, force=False (default resume behavior): properties in `prop`
            already present in the existing output are left as-is (skipped) —
            use this to resume an interrupted/incomplete run.
        update=True, force=True: properties in `prop` are always recomputed and
            overwrite whatever's already there, while properties NOT in `prop`
            are left untouched — use this to refresh specific properties (e.g.
            spectral_centroid/spectral_bandwidth after a dataset fix) without
            rerunning everything else.

        Saves:
            permutation_baseline_nonlinear_clusters.json with structure:
            {feature: {section: {cluster_id: {same fields as layer version}}}}
        """
        output_dir = Path(output_dir)

        clustering_results_path = output_dir / "cross_layer_clustering_results_all_neurons.json"
        if not clustering_results_path.exists():
            raise FileNotFoundError(
                f"Cross-layer clustering results not found at {clustering_results_path}. "
                "Run do_cross_layer_clustering first."
            )
        with open(clustering_results_path) as f:
            clustering_results = json.load(f)

        out_path = output_dir / "permutation_baseline_nonlinear_clusters.json"
        if update and out_path.exists():
            with open(out_path) as f:
                results = json.load(f)
        else:
            results = {}

        print("\n" + "=" * 60)
        print("Nonlinear-Probe Permutation Baseline — Clusters")
        print("=" * 60)
        print(f"  n_permutations={n_permutations}, n_folds={n_folds}, n_epochs={n_epochs}")
        print("  hidden_dim scaled per cluster as min(64, max(8, n_neurons // 8))")

        # ---- Build valid-index / property-value arrays per feature (once) ----
        feature_samples = {}
        for property_key in prop:
            valid_indices, property_values = self.get_balanced_sample(property_key, n=500)
            if len(property_values) < n_folds * 2:
                print(f"  {property_key}: not enough samples ({len(property_values)}), skipping.")
                continue
            feature_samples[property_key] = (valid_indices, property_values)
            print(f"  {property_key}: {len(property_values)} valid samples (balanced)")

        for section_name in ['early', 'middle', 'late']:
            if section_name not in clustering_results:
                continue

            section_data = clustering_results[section_name]
            neuron_layer_map = section_data['neuron_layer_map']
            cluster_labels = np.array(section_data['cluster_labels'])
            n_clusters = section_data['n_clusters']

            print(f"\n{'='*60}\nSection: {section_name}  ({n_clusters} clusters)\n{'='*60}")

            # Reconstruct full section activations [all_samples, all_neurons]
            all_neuron_acts = []
            for layer_name, neuron_idx in neuron_layer_map:
                all_neuron_acts.append(
                    self.activation_records[layer_name].activations[:, neuron_idx]
                )
            section_activations = np.column_stack(all_neuron_acts)  # [N, total_neurons]

            section_results = results.get(section_name, {})

            for cluster_id in range(n_clusters):
                cluster_key = f"cluster_{cluster_id}"
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_acts_full = section_activations[:, cluster_indices].astype(np.float32)
                n_channels = cluster_acts_full.shape[1]
                cluster_hidden_dim = min(64, max(8, n_channels // 8))
                print(f"\n  Cluster {cluster_id}  ({n_channels} neurons, hidden_dim={cluster_hidden_dim})")

                cluster_results = section_results.get(cluster_key, {})

                for property_key, (valid_indices, property_values) in feature_samples.items():
                    if update and not force and property_key in cluster_results:
                        continue
                    acts = cluster_acts_full[valid_indices]  # [N_valid, n_channels]
                    n_samples = len(property_values)

                    obs_r2_folds = self._train_probe_cv(
                        X=acts, y=property_values,
                        hidden_dim=cluster_hidden_dim, n_folds=n_folds,
                        n_epochs=n_epochs, lr=learning_rate,
                        batch_size=batch_size,
                        early_stopping_patience=early_stopping_patience,
                        random_state=random_state,
                    )
                    observed_r2 = float(np.mean(obs_r2_folds))
                    observed_r2_std = float(np.std(obs_r2_folds))

                    linear_observed_r2 = None
                    if include_linear_baseline:
                        lin_folds = self._train_probe_cv(
                            X=acts, y=property_values,
                            hidden_dim=0, n_folds=n_folds,
                            n_epochs=n_epochs, lr=learning_rate,
                            batch_size=batch_size,
                            early_stopping_patience=early_stopping_patience,
                            random_state=random_state,
                        )
                        linear_observed_r2 = float(np.mean(lin_folds))

                    rng = np.random.default_rng(seed=random_state)
                    shuffled_arrays = [rng.permutation(property_values) for _ in range(n_permutations)]

                    print(f"    [{property_key}] {n_permutations} permutations...", flush=True)
                    null_r2 = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                        delayed(_run_one_probe_permutation)(
                            acts, shuffled, cluster_hidden_dim, n_epochs, learning_rate,
                            batch_size, early_stopping_patience, random_state + i,
                        )
                        for i, shuffled in enumerate(shuffled_arrays)
                    )
                    null_r2 = np.array(null_r2, dtype=np.float32)

                    linear_null_mean_r2 = None
                    if include_linear_baseline:
                        lin_null = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                            delayed(_run_one_probe_permutation)(
                                acts, shuffled, 0, n_epochs, learning_rate,
                                batch_size, early_stopping_patience, random_state + i,
                            )
                            for i, shuffled in enumerate(shuffled_arrays)
                        )
                        linear_null_mean_r2 = float(np.mean(lin_null))

                    null_mean = float(np.mean(null_r2))
                    null_std  = float(np.std(null_r2))
                    null_p95  = float(np.percentile(null_r2, 95))
                    ratio = (observed_r2 / null_mean) if abs(null_mean) > 1e-6 else float('nan')

                    entry = {
                        'n_channels': int(n_channels),
                        'hidden_dim': int(cluster_hidden_dim),
                        'n_samples': int(n_samples),
                        'null_mean_r2': null_mean,
                        'null_std_r2': null_std,
                        'null_p95_r2': null_p95,
                        'observed_r2': observed_r2,
                        'observed_r2_std': observed_r2_std,
                        'exceeds_null_p95': bool(observed_r2 > null_p95),
                        'delta_r2': observed_r2 - null_mean,
                        'ratio': ratio,
                    }
                    if include_linear_baseline:
                        entry['linear_observed_r2'] = linear_observed_r2
                        entry['linear_null_mean_r2'] = linear_null_mean_r2
                        if linear_observed_r2 is not None:
                            entry['nonlinear_gain'] = observed_r2 - linear_observed_r2

                    print(f"      obs R²={observed_r2:.4f}±{observed_r2_std:.4f}, "
                          f"null p95={null_p95:.4f}, Δ={entry['delta_r2']:+.4f}"
                          + (f", lin={linear_observed_r2:.4f}" if include_linear_baseline else ""),
                          flush=True)

                    cluster_results[property_key] = entry

                section_results[cluster_key] = cluster_results

            results[section_name] = section_results

            # Save after each section so progress isn't lost on interruption
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved to {out_path}")
        return results


    def do_permutation_baseline_clusters(
        self,
        output_dir,
        prop: List = ['pitch', 'bpm'],
        n_permutations: int = 1000,
        threshold: float = 0.15,
        update: bool = False,
        force: bool = False,
        use_global_null: bool = True,
    ) -> Dict:
        """
        Spearman permutation baseline on cross-layer neuron clusters.

        For each (section, cluster, property) triple, computes observed per-neuron
        |r| and compares against a null distribution.

        use_global_null=True (default):
            Loads the null stats from permutation_baseline.json (the model-wide
            null built across all layers) rather than rerunning permutations per
            cluster.  Consistent with how layer-level observed |r| is compared
            against the same global null, and much faster.

        use_global_null=False:
            Runs n_permutations shuffles per cluster to build a cluster-specific
            null distribution.

        update=True, force=False (default resume behavior): properties in `prop`
            already present in the existing output are left as-is (skipped) —
            use this to resume an interrupted/incomplete run.
        update=True, force=True: properties in `prop` are always recomputed and
            overwrite whatever's already there, while properties NOT in `prop`
            are still left untouched — use this to refresh specific properties
            (e.g. spectral_centroid/spectral_bandwidth after a dataset fix)
            without rerunning everything else.

        Requires cross_layer_clustering_results_all_neurons.json in output_dir.
        When use_global_null=True also requires permutation_baseline.json.

        Saves:
            permutation_baseline_clusters.json
            {section: {cluster_id: {property: same fields as do_permutation_baseline}}}
        """
        output_dir = Path(output_dir)

        clustering_results_path = output_dir / "cross_layer_clustering_results_all_neurons.json"
        if not clustering_results_path.exists():
            raise FileNotFoundError(
                f"Cross-layer clustering results not found at {clustering_results_path}. "
                "Run do_cross_layer_clustering first."
            )
        with open(clustering_results_path) as f:
            clustering_results = json.load(f)

        global_null = None
        if use_global_null:
            global_null_path = output_dir / "permutation_baseline.json"
            if not global_null_path.exists():
                raise FileNotFoundError(
                    f"permutation_baseline.json not found at {global_null_path}. "
                    "Run do_permutation_baseline first, or use use_global_null=False."
                )
            with open(global_null_path) as f:
                global_null = json.load(f)

        out_path = output_dir / "permutation_baseline_clusters.json"
        if update and out_path.exists():
            with open(out_path) as f:
                results = json.load(f)
        else:
            results = {}

        print("\n" + "=" * 60)
        print("Spearman Permutation Baseline — Clusters")
        print("=" * 60)
        if use_global_null:
            print("  Using global null from permutation_baseline.json")
        else:
            print(f"  n_permutations={n_permutations} (per-cluster null)")
        print(f"  threshold=|r|>{threshold}")

        # Extract valid indices/values per property once (same logic as do_permutation_baseline)
        feature_samples = {}
        for property_key in prop:
            valid_indices, property_values = self.get_balanced_sample(property_key, n=500)

            if len(property_values) < 2:
                print(f"  {property_key}: not enough samples, skipping.")
                continue
            feature_samples[property_key] = (valid_indices, property_values)
            print(f"  {property_key}: {len(property_values)} valid samples (balanced)")

        for section_name in ['early', 'middle', 'late']:
            if section_name not in clustering_results:
                continue

            section_data = clustering_results[section_name]
            neuron_layer_map = section_data['neuron_layer_map']
            cluster_labels = np.array(section_data['cluster_labels'])
            n_clusters = section_data['n_clusters']

            print(f"\n{'='*60}\nSection: {section_name}  ({n_clusters} clusters)\n{'='*60}")

            # Reconstruct full section activations [all_samples, all_neurons]
            all_neuron_acts = []
            for layer_name, neuron_idx in neuron_layer_map:
                all_neuron_acts.append(
                    self.activation_records[layer_name].activations[:, neuron_idx]
                )
            section_activations = np.column_stack(all_neuron_acts)

            section_results = results.get(section_name, {})

            for cluster_id in range(n_clusters):
                cluster_key = f"cluster_{cluster_id}"
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_acts_full = section_activations[:, cluster_indices]
                n_neurons = cluster_acts_full.shape[1]
                print(f"\n  Cluster {cluster_id}  ({n_neurons} neurons)")

                cluster_results = section_results.get(cluster_key, {})

                for property_key, (valid_indices, property_values) in feature_samples.items():
                    if update and not force and property_key in cluster_results:
                        continue

                    acts = cluster_acts_full[valid_indices]  # [n_valid, n_neurons]

                    observed_corrs = np.array([
                        abs(spearmanr(property_values, acts[:, j])[0])
                        for j in range(n_neurons)
                    ])
                    print(f"    [{property_key}] obs mean|r|={observed_corrs.mean():.4f}  "
                          f"max|r|={observed_corrs.max():.4f}  "
                          f"%>{threshold}={100*(observed_corrs>threshold).mean():.1f}%")

                    if use_global_null:
                        if property_key not in global_null:
                            print(f"      [{property_key}] not in global null, skipping.")
                            continue
                        g = global_null[property_key]
                        null_p95_individual = g['null_p95_r']
                        entry = {
                            'use_global_null':            True,
                            'n_neurons':                  int(n_neurons),
                            'threshold':                  threshold,
                            'null_mean_r':                g['null_mean_r'],
                            'null_std_r':                 g['null_std_r'],
                            'null_p95_r':                 null_p95_individual,
                            'null_pct_exceeding':         g['null_pct_exceeding'],
                            'null_pct_exceeding_std':     g['null_pct_exceeding_std'],
                            'observed_mean_r':            float(observed_corrs.mean()),
                            'observed_max_r':             float(observed_corrs.max()),
                            'observed_pct_exceeding':     float(
                                (observed_corrs > threshold).mean() * 100),
                            'observed_pct_exceeding_p95': float(
                                (observed_corrs > null_p95_individual).mean() * 100),
                        }
                        print(f"      null p95={null_p95_individual:.4f}  "
                              f"obs %>p95={entry['observed_pct_exceeding_p95']:.1f}%")
                    else:
                        rng = np.random.default_rng(seed=42)
                        shuffled_arrays = [rng.permutation(property_values)
                                           for _ in range(n_permutations)]

                        print(f"    [{property_key}] running {n_permutations} permutations...",
                              flush=True)
                        null_all_corrs = Parallel(n_jobs=-1, backend='loky', verbose=1)(
                            delayed(_run_one_permutation)(s, {"cluster": acts}, threshold)
                            for s in shuffled_arrays
                        )

                        null_all_corrs = [np.array(c) for c in null_all_corrs]
                        null_mean_r = np.array([c.mean() for c in null_all_corrs])
                        null_exceeding = np.array([(c > threshold).mean() * 100
                                                   for c in null_all_corrs])
                        null_pooled = np.concatenate(null_all_corrs)
                        null_p95_individual = float(np.percentile(null_pooled, 95))
                        print(f"      null p95={null_p95_individual:.4f}  "
                              f"obs %>p95={float((observed_corrs > null_p95_individual).mean() * 100):.1f}%",
                              flush=True)
                        entry = {
                            'use_global_null':            False,
                            'n_permutations':             n_permutations,
                            'n_neurons':                  int(n_neurons),
                            'threshold':                  threshold,
                            'null_mean_r':                float(null_mean_r.mean()),
                            'null_std_r':                 float(null_mean_r.std()),
                            'null_p95_r':                 null_p95_individual,
                            'null_pct_exceeding':         float(null_exceeding.mean()),
                            'null_pct_exceeding_std':     float(null_exceeding.std()),
                            'observed_mean_r':            float(observed_corrs.mean()),
                            'observed_max_r':             float(observed_corrs.max()),
                            'observed_pct_exceeding':     float(
                                (observed_corrs > threshold).mean() * 100),
                            'observed_pct_exceeding_p95': float(
                                (observed_corrs > null_p95_individual).mean() * 100),
                        }

                    cluster_results[property_key] = entry

                section_results[cluster_key] = cluster_results

            results[section_name] = section_results

            # Save after each section so progress isn't lost on interruption
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved Spearman cluster baseline to {out_path}")
        return results


    def _train_probe_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_dim: int,
        n_folds: int,
        n_epochs: int,
        lr: float,
        batch_size: int,
        early_stopping_patience: int,
        random_state: int,
    ) -> List[float]:
        """
        Train a probe with k-fold CV. hidden_dim=0 -> linear probe (no hidden layer).
        Returns list of held-out R^2 scores, one per fold.
        """
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # Pre-compute all fold splits so Parallel can dispatch them independently.
        fold_args = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)
            y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
            y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
            y_test_n = ((y_test - y_mean) / y_std).astype(np.float32)
            fold_args.append((X_train, y_train_n, X_test, y_test_n, random_state + fold_idx))

        return Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(_fit_probe_once)(
                Xtr, ytr, Xte, yte,
                hidden_dim=hidden_dim, n_epochs=n_epochs, lr=lr,
                batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                seed=seed,
            )
            for Xtr, ytr, Xte, yte, seed in fold_args
        )


def _build_probe(in_dim: int, hidden_dim: int) -> nn.Module:
    """hidden_dim=0 -> linear probe; >0 -> one-hidden-layer MLP."""
    if hidden_dim == 0:
        return nn.Linear(in_dim, 1)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def _fit_linear_ridge(X_train, y_train, X_test, y_test, seed) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    model = RidgeCV(alphas=np.logspace(-3, 4, 20))
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return float(max(r2, -1.0)) 

def _fit_probe_once(
    X_train, y_train, X_test, y_test,
    hidden_dim, n_epochs, lr, batch_size,
    early_stopping_patience, seed,
) -> float:
    """Train a probe once on a single train/test split. Returns held-out R^2."""
    from sklearn.metrics import r2_score
    
    if hidden_dim == 0:
        return _fit_linear_ridge(X_train, y_train, X_test, y_test, seed)

    torch.manual_seed(seed)
    device = 'cpu'  # probes are tiny; CPU + joblib parallelism is faster overall

    model = _build_probe(in_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test).to(device)

    n = X_train_t.shape[0]
    best_test_r2 = -np.inf
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = criterion(pred, y_train_t[idx])
            loss.backward()
            optimizer.step()

        # Held-out R^2 for early stopping
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t).cpu().numpy().ravel()
        test_r2 = r2_score(y_test, y_pred)
        test_r2 = max(r2_score(y_test, y_pred), -1.0)
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stopping_patience:
                break

    return float(best_test_r2)


def _run_one_probe_permutation(
    X, shuffled_y, hidden_dim, n_epochs, lr, batch_size,
    early_stopping_patience, seed,
) -> float:
    """
    One permutation: train a probe on shuffled labels using a single
    held-out split (no CV — the permutations themselves provide the
    noise estimate). Returns held-out R^2.
    """
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)

    n = len(shuffled_y)
    perm = rng.permutation(n)
    n_test = max(int(0.2 * n), 1)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = shuffled_y[train_idx], shuffled_y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test_n = ((y_test - y_mean) / y_std).astype(np.float32)

    return _fit_probe_once(
        X_train, y_train_n, X_test, y_test_n,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs, lr=lr,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )


def _run_one_permutation(shuffled, layer_activation_cache, threshold):
    """Worker for parallel permutation baseline — pure numpy, no GPU."""
    from scipy.stats import spearmanr as _sr
    perm_corrs = []
    for acts in layer_activation_cache.values():
        for neuron_idx in range(acts.shape[1]):
            r, _ = _sr(shuffled, acts[:, neuron_idx])
            perm_corrs.append(abs(r))
    perm_corrs = np.array(perm_corrs)
    return perm_corrs