"""
encodec_adapter.py

Fits EnCodec into the existing RAVEActivationAnalyser pipeline with maximal
reuse. Only the model-loading, decoder-location, and forward-pass differ;
hook registration, balanced sampling, clustering, and correlation analysis
are inherited unchanged.

Usage
-----
    from encodec_adapter import EncodecActivationAnalyser, load_encodec

    model = load_encodec("facebook/encodec_32khz")        # music-oriented
    analyser = EncodecActivationAnalyser(model, device="cuda")
    analyser.activate(audio_list, metadata_list, sample_rate=32000)
    # ... then call the same clustering / correlation methods as for RAVE
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from collections import defaultdict
from typing import List

# Reuse the entire RAVE analyser; we only override two methods.
from rave_activation_clustering import (
    RAVEActivationAnalyser,
    ActivationRecord,ActivationHook
)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_encodec(model_id: str = "facebook/encodec_32khz"):
    """
    Load a pretrained EnCodec model via HuggingFace transformers.

    model_id options:
        facebook/encodec_24khz  - general audio (speech-leaning)
        facebook/encodec_32khz  - music-oriented (recommended for your data)

    Returns the EncodecModel in eval mode.
    """
    from transformers import EncodecModel
    model = EncodecModel.from_pretrained(model_id)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────

class EncodecActivationAnalyser(RAVEActivationAnalyser):
    """
    EnCodec drop-in for RAVEActivationAnalyser.

    Reuses: hook system, register_decoder_hooks (Conv1d/ConvTranspose1d match
    EnCodec's decoder), balanced sampling, clustering, correlation analysis,
    and the length-group batching / restore-order logic.

    Overrides: _find_decoder (EnCodec nests its decoder differently) and the
    forward pass inside collect_activations (EnCodec's encode/decode API and
    its required input shape differ from RAVE's).
    """

    def register_decoder_hooks(self, layer_pattern=None, hook_all_leaf_modules=False):
        """
        EnCodec-specific hook registration.

        Hooks the real Conv1d / ConvTranspose1d modules inside EnCodec's
        parametrized conv wrappers — NOT the _WeightNorm parametrization objects
        (whose "output" is a weight tensor, not an activation), and NOT the LSTM
        (the sole recurrent layer, excluded from the convolutional depth
        analysis). The output convolution (single output channel) is also
        skipped, as a 1-channel layer cannot be clustered or correlated
        per-neuron.

        Note: EnCodec applies weight norm via torch.nn.utils.parametrize, so the
        conv modules report as ParametrizedConv1d / ParametrizedConvTranspose1d
        but are still instances of nn.Conv1d / nn.ConvTranspose1d, and their
        forward pass produces the correct activation. Matching by isinstance
        therefore captures the post-convolution activation directly.
        """
        self.remove_hooks()

        decoder = self._find_decoder()
        if decoder is None:
            print("WARNING: could not locate EnCodec decoder.")
            return

        n_skipped_param = 0
        n_skipped_lstm = 0
        n_skipped_output = 0

        for name, module in decoder.named_modules():
            # Skip the weight-norm parametrization machinery entirely.
            # (Hooking these returns the normalized weight tensor, not an activation.)
            if "parametrizations" in name:
                n_skipped_param += 1
                continue

            # Skip the recurrent layer.
            if isinstance(module, nn.LSTM) or "lstm" in name.lower():
                n_skipped_lstm += 1
                continue

            # Hook genuine convolution ops. ParametrizedConv1d / 
            # ParametrizedConvTranspose1d are subclasses of these, so isinstance
            # catches them.
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                # Skip the single-channel output convolution (cannot cluster /
                # per-neuron correlate one channel). Detect via out_channels.
                out_ch = getattr(module, "out_channels", None)
                if out_ch is not None and out_ch <= 1:
                    n_skipped_output += 1
                    continue

                hook = ActivationHook(name)
                handle = module.register_forward_hook(hook)
                self.hooks[name] = hook
                self.hook_handles.append(handle)

        print(f"Registered {len(self.hooks)} conv hooks on EnCodec decoder "
              f"(skipped: {n_skipped_param} parametrization objects, "
              f"{n_skipped_lstm} LSTM, {n_skipped_output} single-channel output conv).")

        if not self.hooks:
            print("WARNING: no conv modules matched in EnCodec decoder. "
                  "Check the decoder structure with print_decoder_structure().")

    # EnCodec's decoder is a stack of Conv1d / ConvTranspose1d / LSTM / ELU,
    # exactly the module types register_decoder_hooks already matches, so we do
    # NOT override register_decoder_hooks. We only point it at the right module.
    def _find_decoder(self) -> nn.Module:
        """
        Locate EnCodec's decoder. In the HF EncodecModel the decoder lives at
        model.decoder (an EncodecDecoder whose .layers is the conv stack).
        """
        # HF EncodecModel exposes .decoder directly
        if hasattr(self.model, "decoder") and isinstance(self.model.decoder, nn.Module):
            print(f"Found EnCodec decoder: 'decoder' "
                  f"({type(self.model.decoder).__name__})")
            return self.model.decoder
        # Fallback to the generic search in the parent
        return super()._find_decoder()

    def collect_activations(self, audio_inputs: List[torch.Tensor],
                            batch_size: int = 4):
        """
        Same structure as the parent (length-group batching, restore-order
        re-permutation, hook reset) but with EnCodec's forward pass.

        The only substantive change from the parent is the model call:
        EnCodec is invoked end-to-end so that the decoder hooks fire, using
        its own encode→quantize→decode path rather than RAVE's
        encode()/decode() with a 128-dim latent slice.
        """
        if len(self.hooks) == 0:
            raise RuntimeError("No hooks registered. Cannot collect activations.")

        for hook in self.hooks.values():
            hook.reset()

        # ---- identical length-group batching to the parent ----
        length_groups = defaultdict(list)
        for idx, audio in enumerate(audio_inputs):
            if audio.dim() == 1:
                length = audio.shape[0]
            elif audio.dim() == 2:
                length = audio.shape[1] if audio.shape[0] <= 2 else audio.shape[0]
            else:
                length = audio.shape[-1]
            length_groups[length].append(idx)

        processing_order = []

        for length in sorted(length_groups.keys()):
            indices = length_groups[length]
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = [audio_inputs[idx] for idx in batch_indices]
                processing_order.extend(batch_indices)

                # EnCodec expects mono (batch, 1, time). Normalise shapes the
                # same way the parent does, then force a single channel.
                processed = []
                for audio in batch:
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)                 # (1, time)
                    elif audio.dim() == 2 and audio.shape[0] > audio.shape[1]:
                        audio = audio.transpose(0, 1)              # (chan, time)
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)    # mono mix
                    processed.append(audio)

                batch_tensor = torch.stack(processed).to(self.device)  # (B,1,T)
                batch_tensor = self._resample_to_encodec_rate(batch_tensor)

                # ---- EnCodec forward pass (the one real difference) ----
                self._encodec_forward(batch_tensor)

        # ---- identical restore-order + record conversion to the parent ----
        print("\nConverting activations to records...")
        restore_order = np.argsort(processing_order)
        for layer_name, hook in self.hooks.items():
            activations = hook.get_activations()
            if len(activations) == 0:
                print(f"  WARNING: No activations captured for layer {layer_name}")
                continue
            activations = activations[restore_order]
            self.activation_records[layer_name] = ActivationRecord(
                layer_name=layer_name,
                activations=activations,
            )

    def _resample_to_encodec_rate(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resample (B, 1, T) audio from the source sample rate (set on
        self.sample_rate by activate()) to the rate the loaded EnCodec
        checkpoint expects (model.config.sampling_rate). EnCodec is sensitive
        to sample rate mismatches (e.g. feeding 44.1kHz audio into the 32kHz
        checkpoint shifts pitch/timing), so this must run before encode().
        """
        target_sr = self.model.config.sampling_rate
        source_sr = getattr(self, "sample_rate", target_sr)
        if source_sr == target_sr:
            return batch_tensor

        if not hasattr(self, "_resampler") or self._resampler_rates != (source_sr, target_sr):
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=source_sr, new_freq=target_sr
            ).to(batch_tensor.device)
            self._resampler_rates = (source_sr, target_sr)

        return self._resampler(batch_tensor)

    def _encodec_forward(self, batch_tensor: torch.Tensor):
        """
        Run EnCodec end-to-end so the decoder hooks fire.

        EnCodec (HF) returns audio_codes from .encode and reconstructs with
        .decode(audio_codes, audio_scales). We run the full path at the model's
        default (max) bandwidth. We only need the side effect of populating the
        hooks, so the returned audio is discarded.
        """
        with torch.no_grad():
            enc = self.model.encode(batch_tensor)        # returns codes + scales
            # HF EncodecModel.encode returns an object/tuple with audio_codes,
            # audio_scales. Support both attribute and tuple forms.
            if hasattr(enc, "audio_codes"):
                codes, scales = enc.audio_codes, enc.audio_scales
            else:
                codes, scales = enc[0], enc[1]
            self.model.decode(codes, scales)             # hooks fire here