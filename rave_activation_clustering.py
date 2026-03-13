

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

    def activate(self, audio_list, metadata_list, sample_rate=48000, min_duration_for_bpm=0.5):
        """
        Activate the analyser with audio and metadata.

        Args:
            audio_list: List of audio tensors
            metadata_list: List of metadata dicts
            sample_rate: Sample rate in Hz (default 48000)
            min_duration_for_bpm: Minimum audio duration (seconds) to keep BPM metadata
        """
        # 3. Register hooks on decoder layers
        print("\n" + "="*60)
        print("Step 2: Registering hooks on decoder layers")
        print("="*60)
        print(f"audio_list {len(audio_list)}")
        print(f"metadata_list {len(metadata_list)}")

        # Clean metadata based on audio duration
        print("\n" + "="*60)
        print("Step 2.5: Cleaning metadata based on audio duration")
        print("="*60)

        cleaned_metadata_list = []
        bpm_removed_count = 0

        for i, (audio, metadata) in enumerate(zip(audio_list, metadata_list)):
            # Get audio duration
            if audio.dim() == 1:
                duration = audio.shape[0] / sample_rate
            elif audio.dim() == 2:
                duration = audio.shape[1] / sample_rate if audio.shape[0] <= 2 else audio.shape[0] / sample_rate
            else:
                duration = audio.shape[-1] / sample_rate

            # Clean metadata
            cleaned_metadata = metadata.copy()

            # Remove BPM if audio is too short to measure it reliably
            if duration < min_duration_for_bpm and 'bpm' in cleaned_metadata:
                del cleaned_metadata['bpm']
                bpm_removed_count += 1

            cleaned_metadata_list.append(cleaned_metadata)

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

        # 4. Generate test stimuli
        print("\n" + "="*60)
        print("Step 3: Generating test stimuli")
        print("="*60)

        self.metadata_list = cleaned_metadata_list
        
        
        # 5. Collect activations
        print("\n" + "="*60)
        print("Step 4: Collecting activations")
        print("="*60)
        self.collect_activations(audio_list, batch_size=8)
        
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
        
        print(f"\nScanning decoder for layers to hook...")
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
                print(f"  ✓ Registered hook on: {name} ({type(module).__name__})")
        
        print(f"\nSummary: Registered {registered} hooks out of {candidates} candidate layers")
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
        else:
            # Verify hooks are actually stored
            print(f"Verifying: self.hooks contains {len(self.hooks)} entries")
            print(f"Verifying: self.hook_handles contains {len(self.hook_handles)} entries")
                
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
        print(f"\nCollecting activations for {len(audio_inputs)} audio samples...")
        print(f"Number of hooks registered: {len(self.hooks)}")
        
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

        # Report length distribution
        if len(length_groups) > 1:
            print(f"  Found {len(length_groups)} different audio lengths:")
            for length, indices in sorted(length_groups.items()):
                print(f"    - {length} samples: {len(indices)} files")

        # Process each length group separately
        total_batches = sum((len(indices) + batch_size - 1) // batch_size
                           for indices in length_groups.values())
        batch_counter = 0

        for length in sorted(length_groups.keys()):
            indices = length_groups[length]
            n_batches_this_length = (len(indices) + batch_size - 1) // batch_size

            for batch_idx, i in enumerate(range(0, len(indices), batch_size)):
                batch_indices = indices[i:i+batch_size]
                batch = [audio_inputs[idx] for idx in batch_indices]
                batch_counter += 1

                print(f"  Batch {batch_counter}/{total_batches} (length={length}, {len(batch)} samples)...", end='')

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

                print(" ✓")

            # Activations are now stored in hooks
        
        print("\n" + "-"*60)
        print("DEBUG: Checking hook state after forward passes")
        print(f"  Number of hooks in self.hooks: {len(self.hooks)}")
        print(f"  Hook names: {list(self.hooks.keys())[:3]}...")
        for name, hook in list(self.hooks.items())[:2]:
            print(f"  Hook '{name}' has {len(hook.activations)} activation batches")
        print("-"*60)
        
        print("\nConverting activations to records...")
        # Convert activations to ActivationRecords
        for layer_name, hook in self.hooks.items():
            activations = hook.get_activations()
            if len(activations) == 0:
                print(f"  WARNING: No activations captured for layer {layer_name}")
                continue
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
        metadata = self.metadata_list

        # Extract property values
        properties = []
        valid_indices = []
        for i, meta in enumerate(metadata):
            if property_key in meta and meta[property_key] is not None:
                properties.append(meta[property_key])
                valid_indices.append(i)

        if len(properties) < 2:
            print(f"  Not enough samples with property '{property_key}'")
            return {}

        properties = np.array(properties)
        activations = activations[valid_indices]
        
        # Compute variance of each neuron across stimuli
        neuron_variances = np.var(activations, axis=0)
        
        # Compute correlation between property and each neuron's activation
        
        correlations = []
        p_values = []
        
        for neuron_idx in range(activations.shape[1]):
            corr, pval = spearmanr(properties, activations[:, neuron_idx])
            correlations.append(np.abs(corr))
            p_values.append(pval)
        
        correlations = np.array(correlations)
        p_values = np.array(p_values)
        
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
        
        # print(f"  Found {len(responsive_neurons)} neurons highly responsive to {property_key}")
        # print(f"  Mean |correlation|: {results['mean_correlation']:.3f}")
        # print(f"  Max |correlation|: {results['max_correlation']:.3f}")
        
        return results  

    def load_correlation(self, output_dir):
        variance_path = Path(output_dir) / "variance_correlation.json"
        with open(variance_path, 'r') as f:
            from_file = json.load(f)
        self.correlations = from_file

    def do_correlation(self, output_dir, prop:List = ['pitch', 'pitch_class', 'bpm']):
        print("\n" + "="*60)
        print("Additional Analysis: Neuron Variance-Property Correlation")
        print("="*60)
        
        variance_results = {}
        for layer_name in self.activation_records.keys():
            layer_results = {}
            for p in prop:
                result = self.compute_neuron_variance_correlation(
                    layer_name, property_key=p
                )
                if result:
                    layer_results[p] = result
            
            if layer_results:
                variance_results[layer_name] = layer_results
        
        # Save variance analysis results
        variance_path = Path(output_dir) / "variance_correlation.json"
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
            print(f"\n--- Analyzing layer: {layer_name} ---")
            
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
            'early': [],    # layers 0-7
            'middle': [],   # layers 8-14
            'late': []      # layers 15+
        }

        # Categorize layers by extracting layer number
        for layer_name in self.activation_records.keys():
            # Extract layer number (assuming format like "net.X.something")
            layer_num = None
            if 'net.' in layer_name:
                parts = layer_name.split('.')
                try:
                    layer_num = int(parts[1])
                except:
                    pass

            if layer_num is not None:
                if layer_num <= 7:
                    layer_sections['early'].append(layer_name)
                elif layer_num <= 14:
                    layer_sections['middle'].append(layer_name)
                else:
                    layer_sections['late'].append(layer_name)

        print(f"\nLayer sections:")
        for section, layers in layer_sections.items():
            print(f"  {section}: {len(layers)} layers")

        results_summary = {}

        # Process each section
        for section_name, layer_names in layer_sections.items():
            if len(layer_names) == 0:
                print(f"\n⚠️  No layers in {section_name} section, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {section_name} section ({len(layer_names)} layers)")
            print(f"{'='*60}")

            # Collect activations from all layers in this section
            all_activations = []
            neuron_layer_map = []  # Track which layer and index each neuron belongs to

            for layer_name in layer_names:
                activations = self.activation_records[layer_name].activations.T
                n_neurons = activations.shape[0]

                # Calculate variance for each neuron across samples
                neuron_variances = np.var(activations, axis=1)

                # Sort by variance and keep top neurons
                sorted_indices = np.argsort(neuron_variances)[::-1]
                cumsum = np.cumsum(neuron_variances[sorted_indices])
                total_var = cumsum[-1]
                threshold_idx = np.searchsorted(cumsum, variance_threshold * total_var)

                # Keep at least some neurons from each layer
                n_keep = max(threshold_idx + 1, min(10, n_neurons))
                top_indices = sorted_indices[:n_keep]

                print(f"  {layer_name}: keeping {n_keep}/{n_neurons} neurons ({n_keep/n_neurons*100:.1f}%)")

                # Add selected neurons
                all_activations.append(activations[top_indices])
                # Store both layer name and original neuron index
                neuron_layer_map.extend([(layer_name, int(idx)) for idx in top_indices])

            # Combine all activations
            combined_activations = np.vstack(all_activations)
            print(f"\n  Total neurons in {section_name}: {combined_activations.shape[0]}")
            print(f"  Total samples: {combined_activations.shape[1]}")

            # Create temporary activation record for combined data to reuse existing functions
            temp_layer_name = f'section_{section_name}'
            temp_record = ActivationRecord(
                layer_name=temp_layer_name,
                activations=combined_activations.T
            )
            self.activation_records[temp_layer_name] = temp_record

            # Use existing cluster_neurons function which handles PCA and clustering
            print(f"\n  Applying PCA and clustering...")
            cluster_result = self.cluster_neurons(
                layer_name=temp_layer_name,
                method='kmeans',
                n_clusters=n_clusters,
                use_pca=True,
                pca_components=pca_components
            )

            # Store cluster results for use with analyze_cluster_properties
            self.cluster_results[temp_layer_name] = cluster_result

            print(f"  Silhouette score: {cluster_result.silhouette_score:.3f}")

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
                    print(f"    Analyzed {meta_key} properties")

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
        results_path = output_dir / "cross_layer_clustering_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n✓ Saved cross-layer clustering results to {results_path}")

    def do_cross_layer_correlation(self, output_dir, prop: List = ['pitch', 'pitch_class', 'bpm']):
        """
        Perform correlation analysis on cross-layer clusters.
        Reuses compute_neuron_variance_correlation by creating temporary activation records for each cluster.
        Reconstructs section activations from original layer records and neuron_layer_map.

        Args:
            output_dir: Directory to save results
            prop: List of properties to correlate with
        """
        print("\n" + "="*60)
        print("Cross-Layer Cluster Correlation Analysis")
        print("="*60)

        # Load cross-layer clustering results to get neuron_layer_map and cluster_labels
        output_dir = Path(output_dir)
        clustering_results_path = output_dir / "cross_layer_clustering_results.json"

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

            print(f"\n{'='*60}")
            print(f"Analyzing {section_name} section clusters")
            print(f"{'='*60}")

            section_data = clustering_results[section_name]
            neuron_layer_map = section_data['neuron_layer_map']
            cluster_labels = np.array(section_data['cluster_labels'])
            n_clusters = section_data['n_clusters']

            # Reconstruct section activations from original layer records using neuron_layer_map
            print(f"  Reconstructing section activations from {len(neuron_layer_map)} neurons...")

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
            print(f"  Reconstructed section activations shape: {section_activations.shape}")

            section_correlation_results = {}

            # Analyze each cluster
            for cluster_id in range(n_clusters):
                print(f"\n  Cluster {cluster_id}:")

                # Get neurons belonging to this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_neuron_indices = np.where(cluster_mask)[0]
                n_neurons_in_cluster = len(cluster_neuron_indices)

                print(f"    Neurons in cluster: {n_neurons_in_cluster}")

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
                        print(f"      {p}: mean_corr={result['mean_correlation']:.3f}, "
                              f"n_responsive={result['n_responsive_neurons']}")

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
        results_path = output_dir / "cross_layer_cluster_correlation.json"
        with open(results_path, 'w') as f:
            json.dump(correlation_results, f, indent=2)

        print(f"\n✓ Saved cross-layer cluster correlation results to {results_path}")

        return correlation_results

    def do_permutation_baseline(
        self,
        output_dir,
        prop: List = ['pitch', 'bpm'],
        n_permutations: int = 1000,
        threshold: float = 0.15
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

        Saves:
            permutation_baseline.json  with per-property:
              - null_mean, null_std       (mean/std of null |r| distribution)
              - pct_exceeding_threshold   (% of null correlations > threshold)
              - p95_threshold             (95th-percentile of null distribution)
              - observed_mean, observed_max  (real data, for comparison)
        """
        print("\n" + "="*60)
        print("Permutation Baseline Analysis")
        print("="*60)
        print(f"  n_permutations={n_permutations}, threshold=|r|>{threshold}")

        output_dir = Path(output_dir)
        results = {}

        for property_key in prop:
            print(f"\n--- {property_key} ---")

            # Extract aligned (activation_idx, property_value) pairs once
            valid_indices, property_values = [], []
            for i, meta in enumerate(self.metadata_list):
                if property_key in meta and meta[property_key] is not None:
                    valid_indices.append(i)
                    property_values.append(meta[property_key])

            if len(property_values) < 2:
                print(f"  Not enough samples, skipping.")
                continue

            property_values = np.array(property_values)
            n_samples = len(property_values)

            # Collect all per-neuron |r| from every layer (observed)
            observed_corrs = []
            layer_activation_cache = {}
            for layer_name, record in self.activation_records.items():
                acts = record.activations[valid_indices]   # [samples, neurons]
                layer_activation_cache[layer_name] = acts
                for neuron_idx in range(acts.shape[1]):
                    r, _ = _spearmanr(property_values, acts[:, neuron_idx])
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
            null_all_corrs = Parallel(n_jobs=-1, backend='loky', verbose=10)(
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

        out_path = output_dir / "permutation_baseline.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved permutation baseline to {out_path}")
        return results

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
        n_components: int = 50,
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
        
        # Fit PCA
        pca_full = PCA(random_state=42)
        pca_full.fit(activations)
        
        # Find components for 95% and 99% variance
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_95 = np.argmax(cumvar >= 0.95) + 1
        n_99 = np.argmax(cumvar >= 0.99) + 1
        print(f"  95% variance: {n_95} components")
        print(f"  99% variance: {n_99} components")
        
        pca = PCA(n_components=n_95, random_state=42)
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
        print(f"pca_components {pca_components}")
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
        
        print(f"silhouette score {score}")
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
        print(f"Clustered {layer_name} into {n_clusters} groups")
        
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
