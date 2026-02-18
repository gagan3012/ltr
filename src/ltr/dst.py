"""
Distributional Semantics Tracing (DST)
=======================================

Implements the DST method for tracing semantic failures in decoder-only
transformers. DST produces **layer-wise semantic maps** — compact weighted
directed graphs whose nodes are human-readable concepts most compatible with
the residual stream at a given layer and whose edges summarise causal
dependencies among those concepts under the model's learned representations.

Core pipeline
-------------
1. **Concept scores** — project the residual stream at the *answer position*
   through the unembedding matrix:  s^ℓ(v; i*) = <U_v, h_{i*}^ℓ>.
2. **Top-K node selection** with subword merging so that nodes correspond to
   human-readable words rather than subword fragments.
3. **Causal edge weights** — for each node v, identify the most influential
   prompt position, minimally corrupt it, re-run the model, and measure the
   probability drop on every other node w:
     Ω^ℓ(v⇒w) = P(t_w | x) − P(t_w | x̃_{p^ℓ(v)}).
4. **Contextual Alignment Score (CAS)** — fraction of total absolute cosine
   alignment assigned to context-consistent concepts, with operational layer
   markers: prediction onset, semantic inversion, and commitment.

Data classes
------------
- ``SemanticMapNode`` — a concept node with word, score, and affinity.
- ``SemanticMapEdge`` — a directed causal edge v ⇒ w.
- ``SemanticMap``     — per-layer concept graph (nodes + edges).
- ``CASTrace``        — CAS values across depth with layer markers.
- ``DSTResult``       — full analysis output.

Usage
-----
>>> from ltr.dst import DistributionalSemanticsTracer
>>> tracer = DistributionalSemanticsTracer(model, tokenizer)
>>> result = tracer.run_analysis(
...     prompt,
...     context_words=["river", "water"],
...     noncontext_words=["money", "finance"],
... )
>>> tracer.plot_dst_summary(result, context_words=["river"], noncontext_words=["money"])
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from baukit import TraceDict
except ImportError:
    TraceDict = None

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SemanticMapNode:
    """A concept node in the layer-wise semantic map."""
    word: str                # Human-readable surface word
    token_ids: List[int]     # Constituent vocabulary token ids
    score: float             # Compatibility score  s^ℓ(v; i*)
    affinity: float          # Cosine alignment    a^ℓ(v) = cos(h, e(v))


@dataclass
class SemanticMapEdge:
    """A directed causal edge  v ⇒ w."""
    source: str              # Word of source node
    target: str              # Word of target node
    weight: float            # Ω^ℓ(v ⇒ w) — probability drop
    source_position: int     # Influential prompt position  p^ℓ(v)


@dataclass
class SemanticMap:
    """Per-layer semantic map  G^ℓ = (V^ℓ, E^ℓ)."""
    layer: int
    nodes: List[SemanticMapNode]
    edges: List[SemanticMapEdge]

    def node_words(self) -> List[str]:
        return [n.word for n in self.nodes]

    def edge_tuples(self) -> List[Tuple[str, str, float]]:
        return [(e.source, e.target, e.weight) for e in self.edges]


@dataclass
class CASTrace:
    """Contextual Alignment Score trace across layers."""
    cas_values: List[float]                 # CAS^ℓ for each layer
    onset_layer: Optional[int] = None       # Prediction onset   (green dot)
    inversion_layer: Optional[int] = None   # Semantic inversion (yellow dot)
    commitment_layer: Optional[int] = None  # Commitment          (red dot)


@dataclass
class DSTResult:
    """Container for complete DST analysis results."""
    semantic_maps: Dict[int, SemanticMap]    # layer → SemanticMap
    cas_trace: CASTrace                      # CAS across depth
    next_token_probs: Dict[str, float]       # word → probability at answer pos
    concept_importance: Dict[int, float]     # layer → aggregate node score
    generated_text: str = ""                  # greedy continuation from the model
    # Legacy compatibility fields
    patched_representations: Dict[str, Any] = field(default_factory=dict)
    spurious_spans: List[Dict] = field(default_factory=list)
    semantic_drift_trajectory: Dict = field(default_factory=dict)
    intervention_results: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Main tracer
# ---------------------------------------------------------------------------


class DistributionalSemanticsTracer:
    """
    Implements Distributional Semantics Tracing (DST).

    Given a frozen decoder-only transformer, DST produces a layer-wise
    semantic map by:

    1.  Projecting the residual stream at the *answer position* through
        the unembedding matrix to obtain per-vocabulary compatibility scores.
    2.  Selecting top-K concept nodes (with subword merging).
    3.  Computing directed causal edges via minimal prompt corruption.
    4.  Computing the Contextual Alignment Score (CAS) and deriving
        operational layer markers (onset, inversion, commitment).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "auto",
        layer_prefix: str = "model.layers.",
        batch_size: int = 1,
    ):
        """
        Parameters
        ----------
        model : PreTrainedModel
            A frozen decoder-only transformer.
        tokenizer : PreTrainedTokenizer
            The model's tokenizer.
        device : str
            ``"auto"`` selects CUDA when available.
        layer_prefix : str
            Prefix for residual-stream layer names used by baukit.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.layer_prefix = layer_prefix

        self.n_layers = self._detect_num_layers()
        self.layer_names = [
            f"{self.layer_prefix}{i}" for i in range(self.n_layers)
        ]

        # Cached unembedding matrix  U ∈ R^{|V| × d}
        self._unembed_weight: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_num_layers(self) -> int:
        """Detect the number of transformer layers from model config."""
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for attr in ("num_hidden_layers", "n_layer", "n_layers"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        return 12  # fallback

    @torch.no_grad()
    def _get_unembed_matrix(self) -> torch.Tensor:
        """
        Return the unembedding matrix  U ∈ R^{|V| × d}.

        Handles both tied and untied weight configurations.
        """
        if self._unembed_weight is not None:
            return self._unembed_weight

        if hasattr(self.model, "lm_head"):
            W = self.model.lm_head.weight.detach()
        elif hasattr(self.model, "get_output_embeddings"):
            oe = self.model.get_output_embeddings()
            W = oe.weight.detach() if hasattr(oe, "weight") else None
        else:
            W = None

        if W is None:
            ie = self.model.get_input_embeddings()
            W = ie.weight.detach()

        self._unembed_weight = W.to(self.device)
        return self._unembed_weight

    def _encode(self, text: str) -> dict:
        """Tokenize text and move tensors to the model device."""
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _get_residual_stream(
        self,
        tokens: dict,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Run a forward pass and collect residual-stream vectors at
        specified layers.

        Returns
        -------
        dict : {layer_index: tensor of shape (batch, seq, d)}
        """
        if layers is None:
            layers = list(range(self.n_layers))
        names = [self.layer_names[l] for l in layers]

        with TraceDict(self.model, names) as traces:
            self.model(**tokens)
            result = {}
            for l, n in zip(layers, names):
                out = traces[n].output
                result[l] = (
                    out[0].detach() if isinstance(out, tuple) else out.detach()
                )
        return result

    @torch.no_grad()
    def _forward_logits(self, tokens: dict) -> torch.Tensor:
        """Full forward pass returning logits (batch, seq, |V|)."""
        return self.model(**tokens).logits

    # ==================================================================
    # Step 1: Concept scores  s^ℓ(v; i*)
    # ==================================================================

    @torch.no_grad()
    def compute_concept_scores(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. (1):  s^ℓ(v; i*) = ⟨U_v, h_{i*}^ℓ⟩

        Parameters
        ----------
        hidden : Tensor of shape (d,)
            Residual-stream vector at the answer position.

        Returns
        -------
        scores : Tensor of shape (|V|,)
            Compatibility score for every vocabulary item.
        """
        U = self._get_unembed_matrix()  # (|V|, d)
        return U @ hidden               # (|V|,)

    # ==================================================================
    # Step 2: Top-K concept node selection with subword merging
    # ==================================================================

    def _is_subword_continuation(self, token_id: int) -> bool:
        """
        Detect whether a vocabulary token is a subword continuation
        (i.e. never starts a new word).

        Works across BPE (GPT-2 style "Ġ"), SentencePiece ("▁"), and
        tokenizers that prefix whole-word tokens with a space.
        """
        # Use convert_ids_to_tokens to get the raw vocab string before
        # any detokenization normalisation.
        raw = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        if raw is None:
            return True
        # Whole-word tokens start with "Ġ" (GPT-2/RoBERTa BPE),
        # "▁" (SentencePiece/Gemma/Llama), or a leading space.
        # If none of these are present, the token is a continuation piece.
        if raw.startswith("Ġ") or raw.startswith("▁") or raw.startswith(" "):
            return False
        # Special tokens, single punctuation, digits, etc. are standalone.
        if len(raw) <= 1 or raw.startswith("<") or raw.startswith("["):
            return False
        return True

    def _clean_token_word(self, token_id: int) -> str:
        """Decode a single token id to a clean surface string."""
        raw = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        if raw is None:
            return ""
        # Strip SentencePiece / GPT-2 word-boundary markers.
        return raw.lstrip("Ġ▁ ").strip()

    def _merge_subwords_vocab(
        self,
        token_ids: List[int],
        scores: torch.Tensor,
    ) -> List[Tuple[str, List[int], float]]:
        """
        Group vocabulary tokens by decoded surface word.

        Unlike sequential-text subword merging, here the token_ids come
        from an unordered top-K selection over the full vocabulary.
        Tokens that decode to the same cleaned surface form are merged
        (scores summed).  Subword continuation pieces that appear without
        a matching head are kept as standalone entries.

        Returns
        -------
        list of (word, [token_ids], aggregated_score)
        """
        word_groups: Dict[str, Tuple[List[int], float]] = {}
        scores_list = scores.tolist() if isinstance(scores, torch.Tensor) else list(scores)

        for tid, sc in zip(token_ids, scores_list):
            word = self._clean_token_word(tid)
            if not word:
                continue
            # Normalise to lower-case for grouping so "The" and "the"
            # map to the same concept node.
            key = word.lower()
            if key in word_groups:
                existing_ids, existing_sc = word_groups[key]
                word_groups[key] = (existing_ids + [tid], existing_sc + sc)
            else:
                word_groups[key] = ([tid], sc)

        # Use the cleanest decoded form as display name
        merged = []
        for key, (tids, agg_sc) in word_groups.items():
            # Pick the surface form from the first (highest-scoring) token
            display = self._clean_token_word(tids[0])
            merged.append((display, tids, agg_sc))

        return merged

    @torch.no_grad()
    def select_concept_nodes(
        self,
        hidden: torch.Tensor,
        K: int = 30,
        ensure_tokens: Optional[List[int]] = None,
    ) -> List[SemanticMapNode]:
        """
        Eq. (2):  V^ℓ = TopK({s^ℓ(v; i*)}, K)

        With vocabulary-level deduplication so that nodes correspond to
        distinct human-readable concepts rather than subword fragments.

        Parameters
        ----------
        hidden : Tensor (d,)
            Residual stream at the answer position.
        K : int
            Number of concept nodes to return.
        ensure_tokens : list of int or None
            Token ids that *must* appear in the returned set (gold / foil).
        """
        scores = self.compute_concept_scores(hidden)  # (|V|,)
        U = self._get_unembed_matrix()

        # Retrieve a surplus of top candidates to allow for deduplication
        n_candidates = min(K * 5, scores.shape[0])
        _, topk_ids = torch.topk(scores, n_candidates)
        candidate_ids = topk_ids.tolist()

        # Guarantee required tokens are present
        if ensure_tokens:
            for tid in ensure_tokens:
                if tid not in candidate_ids:
                    candidate_ids.append(tid)

        # Group tokens that decode to the same surface word
        merged = self._merge_subwords_vocab(
            candidate_ids, scores[candidate_ids]
        )
        merged.sort(key=lambda x: x[2], reverse=True)
        merged = merged[:K]

        # Build nodes with affinity  a^ℓ(v) = cos(h, e(v))
        h_norm = F.normalize(hidden.unsqueeze(0), dim=-1).squeeze(0)
        nodes: List[SemanticMapNode] = []
        for word, tids, agg_score in merged:
            e_v = U[tids].mean(dim=0)
            e_v_norm = F.normalize(e_v.unsqueeze(0), dim=-1).squeeze(0)
            affinity = float(h_norm @ e_v_norm)
            nodes.append(
                SemanticMapNode(
                    word=word,
                    token_ids=tids,
                    score=agg_score,
                    affinity=affinity,
                )
            )
        return nodes

    # ==================================================================
    # Step 3: Causal edge weights  Ω^ℓ(v ⇒ w)
    # ==================================================================

    def _find_influential_position(
        self,
        residuals: torch.Tensor,
        token_ids: List[int],
        seq_len: int,
    ) -> int:
        """
        For concept v with ``token_ids``, find the prompt position
        p^ℓ(v) whose residual state yields the highest unembedding score
        for v.

        Parameters
        ----------
        residuals : Tensor (seq, d)
        seq_len   : int — length of the prompt.
        """
        U = self._get_unembed_matrix()
        e_v = U[token_ids].mean(dim=0)           # (d,)
        pos_scores = residuals[:seq_len] @ e_v    # (seq_len,)
        return int(pos_scores.argmax().item())

    @torch.no_grad()
    def compute_causal_edges(
        self,
        tokens: dict,
        nodes: List[SemanticMapNode],
        layer: int,
        answer_pos: int,
        residuals: torch.Tensor,
        corruption_token: Optional[int] = None,
        max_edges: int = 50,
    ) -> List[SemanticMapEdge]:
        """
        Eq. (4):
            Ω^ℓ(v ⇒ w) = P(t_w | x) − P(t_w | x̃_{p^ℓ(v)})

        For each node v, find the most influential prompt position p^ℓ(v),
        corrupt that single token, re-run the model, and measure the drop
        in probability assigned to every other node w.

        Parameters
        ----------
        tokens : dict
            Tokenized prompt.
        nodes  : list of SemanticMapNode
        layer  : int
        answer_pos : int
        residuals  : Tensor (seq, d) — residual stream at the given layer.
        corruption_token : int or None
            Token id used for the minimal corruption (defaults to '.').
        max_edges : int
            Keep at most this many edges (by absolute weight).
        """
        if corruption_token is None:
            corruption_token = self.tokenizer.encode(
                ".", add_special_tokens=False
            )[0]

        input_ids = tokens["input_ids"].clone()
        seq_len = input_ids.shape[1]

        # Original next-token distribution at the answer position
        orig_logits = self._forward_logits(tokens)
        orig_probs = F.softmax(orig_logits[0, answer_pos], dim=-1)

        edges: List[SemanticMapEdge] = []
        _corruption_cache: Dict[int, torch.Tensor] = {}

        for v_node in nodes:
            p_v = self._find_influential_position(
                residuals, v_node.token_ids, seq_len
            )
            if p_v == answer_pos:
                continue  # skip — influential pos is the answer itself

            # Corrupted forward pass (cached by position)
            if p_v not in _corruption_cache:
                corrupt_ids = input_ids.clone()
                corrupt_ids[0, p_v] = corruption_token
                corrupt_tokens = {
                    k: (corrupt_ids if k == "input_ids" else v)
                    for k, v in tokens.items()
                }
                corrupt_logits = self._forward_logits(corrupt_tokens)
                _corruption_cache[p_v] = F.softmax(
                    corrupt_logits[0, answer_pos], dim=-1
                )

            corrupt_probs = _corruption_cache[p_v]

            for w_node in nodes:
                if w_node.word == v_node.word:
                    continue
                p_w_orig = orig_probs[w_node.token_ids].sum().item()
                p_w_corr = corrupt_probs[w_node.token_ids].sum().item()
                omega = p_w_orig - p_w_corr
                if abs(omega) > 1e-6:
                    edges.append(
                        SemanticMapEdge(
                            source=v_node.word,
                            target=w_node.word,
                            weight=omega,
                            source_position=p_v,
                        )
                    )

        edges.sort(key=lambda e: abs(e.weight), reverse=True)
        return edges[:max_edges]

    # ==================================================================
    # Build semantic map for a single layer
    # ==================================================================

    @torch.no_grad()
    def build_semantic_map(
        self,
        tokens: dict,
        layer: int,
        answer_pos: int,
        K: int = 30,
        ensure_tokens: Optional[List[int]] = None,
        corruption_token: Optional[int] = None,
        compute_edges: bool = True,
    ) -> SemanticMap:
        """
        Construct the full semantic map G^ℓ for one layer.

        Parameters
        ----------
        tokens : dict          — tokenized prompt
        layer  : int           — layer index
        answer_pos : int       — token position used for prediction
        K      : int           — number of concept nodes
        ensure_tokens : list   — token ids that must appear in the map
        corruption_token : int — token used for minimal corruption
        compute_edges : bool   — whether to compute causal edges (Step 3)
        """
        residuals = self._get_residual_stream(tokens, [layer])[layer]
        h = residuals[0, answer_pos]  # (d,)

        nodes = self.select_concept_nodes(
            h, K=K, ensure_tokens=ensure_tokens
        )

        edges: List[SemanticMapEdge] = []
        if compute_edges:
            edges = self.compute_causal_edges(
                tokens,
                nodes,
                layer,
                answer_pos,
                residuals[0],
                corruption_token=corruption_token,
            )

        return SemanticMap(layer=layer, nodes=nodes, edges=edges)

    # ==================================================================
    # CAS: Contextual Alignment Score
    # ==================================================================

    def _get_word_embedding(self, word: str) -> torch.Tensor:
        """
        Get the embedding vector for a word.

        If the word tokenizes into multiple subword pieces, average
        their embedding vectors to obtain a single representation.

        Returns
        -------
        Tensor of shape (d,)
        """
        tids = self.tokenizer.encode(word, add_special_tokens=False)
        if not tids:
            # Fallback: try with a space prefix (common for SentencePiece)
            tids = self.tokenizer.encode(" " + word, add_special_tokens=False)
        if not tids:
            raise ValueError(f"Cannot tokenize word: {word!r}")

        U = self._get_unembed_matrix()  # (|V|, d)
        vecs = U[tids]                  # (n_pieces, d)
        return vecs.mean(dim=0)         # (d,)

    @torch.no_grad()
    def compute_cas(
        self,
        tokens: dict,
        answer_pos: int,
        context_words: List[str],
        noncontext_words: List[str],
        layers: Optional[List[int]] = None,
        K: int = 30,
    ) -> CASTrace:
        """
        Eq. (6)–(7):

            a^ℓ(v) = cos(h_{i*}^ℓ, e(v))

            CAS^ℓ = Σ_{v ∈ V_ctx} |a^ℓ(v)|
                    / (Σ_{v ∈ V_ctx} |a^ℓ(v)| + Σ_{v ∈ V_nonctx} |a^ℓ(v)|)

        Computes CAS *directly* from the cosine similarity between the
        residual stream and each context / noncontext word embedding.
        This does NOT rely on top-K concept selection.

        Parameters
        ----------
        context_words    : words compatible with the *intended* interpretation.
        noncontext_words : words compatible with the *competing* interpretation.
        K : int
            Unused (kept for API compatibility).
        """
        if layers is None:
            layers = list(range(self.n_layers))

        residuals = self._get_residual_stream(tokens, layers)

        # Pre-compute embedding vectors for each word
        ctx_embeddings: List[torch.Tensor] = []
        for w in context_words:
            try:
                ctx_embeddings.append(self._get_word_embedding(w))
            except ValueError:
                warnings.warn(f"Skipping untokenizable context word: {w!r}")

        nonctx_embeddings: List[torch.Tensor] = []
        for w in noncontext_words:
            try:
                nonctx_embeddings.append(self._get_word_embedding(w))
            except ValueError:
                warnings.warn(
                    f"Skipping untokenizable noncontext word: {w!r}"
                )

        if not ctx_embeddings and not nonctx_embeddings:
            warnings.warn("No valid context/noncontext embeddings — CAS undefined.")
            return CASTrace(
                cas_values=[0.5] * len(layers),
                onset_layer=None,
                inversion_layer=None,
                commitment_layer=None,
            )

        # Stack for vectorised cosine similarity
        # ctx_mat: (n_ctx, d),  nonctx_mat: (n_nonctx, d)
        ctx_mat = (
            torch.stack(ctx_embeddings)
            if ctx_embeddings
            else torch.zeros(0, self._get_unembed_matrix().shape[1], device=self.device)
        )
        nonctx_mat = (
            torch.stack(nonctx_embeddings)
            if nonctx_embeddings
            else torch.zeros(0, self._get_unembed_matrix().shape[1], device=self.device)
        )

        cas_values: List[float] = []
        for l in layers:
            h = residuals[l][0, answer_pos]  # (d,)
            h_norm = F.normalize(h.unsqueeze(0), dim=-1)  # (1, d)

            # Cosine affinities  a^ℓ(v) = cos(h, e(v))
            if ctx_mat.shape[0] > 0:
                ctx_normed = F.normalize(ctx_mat, dim=-1)        # (n_ctx, d)
                ctx_cos = (h_norm @ ctx_normed.T).squeeze(0)     # (n_ctx,)
                ctx_score = float(ctx_cos.abs().sum())
            else:
                ctx_score = 0.0

            if nonctx_mat.shape[0] > 0:
                nonctx_normed = F.normalize(nonctx_mat, dim=-1)  # (n_nonctx, d)
                nonctx_cos = (h_norm @ nonctx_normed.T).squeeze(0)
                nonctx_score = float(nonctx_cos.abs().sum())
            else:
                nonctx_score = 0.0

            total = ctx_score + nonctx_score
            cas = ctx_score / total if total > 0.0 else 0.5
            cas_values.append(cas)

        # Derive operational markers
        onset = self._find_onset(cas_values)
        inversion = self._find_inversion(cas_values, threshold=0.8)
        commitment = self._find_commitment(cas_values, inversion, threshold=0.5)

        return CASTrace(
            cas_values=cas_values,
            onset_layer=onset,
            inversion_layer=inversion,
            commitment_layer=commitment,
        )

    # ------------------------------------------------------------------
    # Operational layer markers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_onset(
        cas: List[float],
        tolerance: float = 0.02,
        window: int = 2,
    ) -> Optional[int]:
        """
        **Prediction onset** (green dot):
        Earliest layer where CAS begins a sustained decline relative to
        the immediately preceding layer.
        """
        for i in range(1, len(cas)):
            if cas[i] < cas[i - 1] - tolerance:
                sustained = all(
                    cas[min(i + j, len(cas) - 1)] <= cas[i] + tolerance
                    for j in range(1, window + 1)
                )
                if sustained:
                    return i
        return None

    @staticmethod
    def _find_inversion(
        cas: List[float],
        threshold: float = 0.8,
    ) -> Optional[int]:
        """
        **Semantic inversion** (yellow dot):
        First layer where CAS falls below a fixed threshold.
        """
        for i, c in enumerate(cas):
            if c < threshold:
                return i
        return None

    @staticmethod
    def _find_commitment(
        cas: List[float],
        inversion: Optional[int],
        threshold: float = 0.5,
        min_persist: int = 3,
    ) -> Optional[int]:
        """
        **Commitment** (red dot):
        First layer after inversion beyond which CAS remains persistently
        low through the remaining depth.
        """
        if inversion is None:
            return None
        for i in range(inversion, len(cas)):
            if cas[i] < threshold:
                remaining = cas[i:]
                if all(c < threshold for c in remaining[:min_persist]):
                    return i
        return None

    # ==================================================================
    # Full analysis pipeline
    # ==================================================================

    @torch.no_grad()
    def run_analysis(
        self,
        prompt: str,
        context_words: Optional[List[str]] = None,
        noncontext_words: Optional[List[str]] = None,
        answer_pos: Optional[int] = None,
        K: int = 30,
        map_layers: Optional[List[int]] = None,
        compute_edges: bool = True,
        ensure_tokens: Optional[List[int]] = None,
        corruption_token: Optional[int] = None,
        cas_layers: Optional[List[int]] = None,
        # Legacy compatibility parameters
        factual_prompt: Optional[str] = None,
        concept_examples: Optional[List[str]] = None,
        hallucinated_output: Optional[str] = None,
        run_intervention: bool = False,
        enhanced_viz: bool = True,
        num_layers: int = 4,
    ) -> DSTResult:
        """
        Run the complete DST analysis pipeline.

        Parameters
        ----------
        prompt : str
            The prompt to analyse.
        context_words : list of str, optional
            Words compatible with the intended interpretation (for CAS).
        noncontext_words : list of str, optional
            Words compatible with the competing interpretation (for CAS).
        answer_pos : int, optional
            Token position used for next-token prediction (defaults to
            the last position).
        K : int
            Number of concept nodes per semantic map.
        map_layers : list of int, optional
            Layers at which to build full semantic maps.  Defaults to a
            representative subset (~6 evenly spaced layers).
        compute_edges : bool
            Whether to compute causal edges (Step 3).  Set False for speed.
        ensure_tokens : list of int, optional
            Token ids that must appear in every concept map (e.g., gold/foil).
        corruption_token : int, optional
            Token id used for minimal corruption in Step 3.
        cas_layers : list of int, optional
            Layers over which to compute CAS (defaults to all layers).

        Returns
        -------
        DSTResult
        """
        tokens = self._encode(prompt)
        seq_len = tokens["input_ids"].shape[1]
        if answer_pos is None:
            answer_pos = seq_len - 1

        # Default map layers: ~6 evenly spaced
        if map_layers is None:
            n_maps = min(6, self.n_layers)
            map_layers = np.linspace(
                0, self.n_layers - 1, n_maps, dtype=int
            ).tolist()

        # ---- Build semantic maps ----
        semantic_maps: Dict[int, SemanticMap] = {}
        for l in tqdm(map_layers, desc="Building semantic maps"):
            sm = self.build_semantic_map(
                tokens,
                layer=l,
                answer_pos=answer_pos,
                K=K,
                ensure_tokens=ensure_tokens,
                corruption_token=corruption_token,
                compute_edges=compute_edges,
            )
            semantic_maps[l] = sm

        # ---- CAS trace ----
        cas_trace = CASTrace(cas_values=[])
        if context_words and noncontext_words:
            cas_trace = self.compute_cas(
                tokens,
                answer_pos=answer_pos,
                context_words=context_words,
                noncontext_words=noncontext_words,
                layers=cas_layers,
                K=K,
            )

        # ---- Next-token probabilities ----
        logits = self._forward_logits(tokens)
        probs = F.softmax(logits[0, answer_pos], dim=-1)
        topk_probs, topk_ids = torch.topk(probs, min(K, probs.shape[0]))
        next_token_probs: Dict[str, float] = {}
        for tid, p in zip(topk_ids.tolist(), topk_probs.tolist()):
            word = self.tokenizer.decode([tid]).strip()
            next_token_probs[word] = p

        # ---- Greedy continuation (short) ----
        gen_ids = self.model.generate(
            tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
            max_new_tokens=30,
            do_sample=False,
        )
        generated_text = self.tokenizer.decode(
            gen_ids[0, seq_len:], skip_special_tokens=True
        ).strip()

        # ---- Concept importance (aggregate score per layer) ----
        concept_importance: Dict[int, float] = {
            l: sum(n.score for n in sm.nodes)
            for l, sm in semantic_maps.items()
        }

        return DSTResult(
            semantic_maps=semantic_maps,
            cas_trace=cas_trace,
            next_token_probs=next_token_probs,
            concept_importance=concept_importance,
            generated_text=generated_text,
        )

    # ==================================================================
    # DSS: Distributional Semantics Strength  (retained for compat)
    # ==================================================================

    @torch.no_grad()
    def compute_dss(
        self,
        prompt: str,
        correct_pathways: List[List[str]],
        incorrect_pathways: List[List[str]],
        layer_idx: int = -1,
    ) -> float:
        """
        Compute the Distributional Semantics Strength (DSS) metric.

        DSS = Σ correct_pathway_strength / (Σ correct + Σ incorrect + ε)
        """
        tokens = self._encode(prompt)
        if layer_idx < 0:
            layer_idx = self.n_layers + layer_idx
        layer_name = self.layer_names[layer_idx]

        with TraceDict(self.model, [layer_name]) as traces:
            self.model(**tokens)
            out = traces[layer_name].output
            layer_act = (
                out[0].detach() if isinstance(out, tuple) else out.detach()
            )

        all_concepts: Set[str] = set()
        for pw in correct_pathways + incorrect_pathways:
            all_concepts.update(pw)

        concept_vecs: Dict[str, torch.Tensor] = {}
        for concept in all_concepts:
            c_toks = self.tokenizer.encode(concept, add_special_tokens=False)
            found = False
            for i in range(
                tokens["input_ids"].shape[1] - len(c_toks) + 1
            ):
                if all(
                    tokens["input_ids"][0, i + j].item() == c_toks[j]
                    for j in range(len(c_toks))
                ):
                    concept_vecs[concept] = layer_act[
                        0, i : i + len(c_toks)
                    ].mean(0)
                    found = True
                    break
            if not found:
                ct = self._encode(concept)
                with TraceDict(self.model, [layer_name]) as ct_tr:
                    self.model(**ct)
                    ct_out = ct_tr[layer_name].output
                    ct_act = (
                        ct_out[0].detach()
                        if isinstance(ct_out, tuple)
                        else ct_out.detach()
                    )
                concept_vecs[concept] = ct_act[0].mean(0)

        def pathway_strength(pathway: List[str]) -> float:
            s = 0.0
            for i in range(len(pathway) - 1):
                a, b = pathway[i], pathway[i + 1]
                if a in concept_vecs and b in concept_vecs:
                    va = concept_vecs[a].cpu().numpy()
                    vb = concept_vecs[b].cpu().numpy()
                    s += abs(float(np.corrcoef(va, vb)[0, 1]))
            return s

        cs = sum(pathway_strength(p) for p in correct_pathways)
        ics = sum(pathway_strength(p) for p in incorrect_pathways)
        return cs / (cs + ics + 1e-10)

    # ==================================================================
    # Visualization
    # ==================================================================

    def plot_semantic_map(
        self,
        smap: SemanticMap,
        ax: Optional[plt.Axes] = None,
        top_n_nodes: int = 15,
        top_n_edges: int = 20,
        node_color_ctx: Optional[Set[str]] = None,
        node_color_nonctx: Optional[Set[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Draw a semantic map as a directed graph.

        Parameters
        ----------
        smap : SemanticMap
            The semantic map to visualise.
        node_color_ctx : set of str
            Node words to colour green (context-consistent).
        node_color_nonctx : set of str
            Node words to colour red (competing interpretation).
        """
        if not _HAS_NX:
            raise ImportError("networkx is required for graph visualization")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        G = nx.DiGraph()

        display_nodes = smap.nodes[:top_n_nodes]
        for n in display_nodes:
            G.add_node(n.word, score=n.score, affinity=n.affinity)

        node_set = {n.word for n in display_nodes}

        display_edges = [
            e
            for e in smap.edges
            if e.source in node_set and e.target in node_set
        ][:top_n_edges]
        for e in display_edges:
            G.add_edge(e.source, e.target, weight=e.weight)

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
            ax.axis("off")
            return fig

        pos = nx.spring_layout(
            G, seed=42, k=2.0 / max(1, math.sqrt(G.number_of_nodes()))
        )

        # Node colours
        ctx_set = node_color_ctx or set()
        nonctx_set = node_color_nonctx or set()
        colours = []
        for nd in G.nodes:
            if nd in ctx_set:
                colours.append("#6fcf97")   # green
            elif nd in nonctx_set:
                colours.append("#eb5757")   # red
            else:
                colours.append("#a0c4ff")   # blue

        sizes = [
            300 + 40 * abs(G.nodes[nd].get("score", 0)) for nd in G.nodes
        ]

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=colours, node_size=sizes, alpha=0.85,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

        if G.number_of_edges() > 0:
            edge_weights = [G[u][v]["weight"] for u, v in G.edges]
            edge_colours = [
                "#27ae60" if w > 0 else "#c0392b" for w in edge_weights
            ]
            edge_widths = [
                0.5 + 3 * min(abs(w), 0.1) / 0.1 for w in edge_weights
            ]
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=edge_colours,
                width=edge_widths,
                alpha=0.6,
                arrows=True,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
            )

        ax.set_title(f"Semantic Map — Layer {smap.layer}", fontsize=12)
        ax.axis("off")

        # Legend
        handles = []
        if ctx_set:
            handles.append(
                mpatches.Patch(color="#6fcf97", label="Context-consistent")
            )
        if nonctx_set:
            handles.append(
                mpatches.Patch(color="#eb5757", label="Competing")
            )
        handles.append(mpatches.Patch(color="#a0c4ff", label="Other"))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)

        return fig

    def plot_cas_trace(
        self,
        cas: CASTrace,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 4),
    ) -> plt.Figure:
        """
        Plot CAS^ℓ across layers with onset / inversion / commitment
        markers.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        layers = list(range(len(cas.cas_values)))
        ax.plot(
            layers, cas.cas_values,
            "o-", color="#2d3436", linewidth=1.5, markersize=4,
        )

        if cas.onset_layer is not None:
            ax.plot(
                cas.onset_layer,
                cas.cas_values[cas.onset_layer],
                "o", color="green", markersize=12,
                label="Prediction onset", zorder=5,
            )
        if cas.inversion_layer is not None:
            ax.plot(
                cas.inversion_layer,
                cas.cas_values[cas.inversion_layer],
                "o", color="#f1c40f", markersize=12,
                label="Semantic inversion", zorder=5,
            )
        if cas.commitment_layer is not None:
            ax.plot(
                cas.commitment_layer,
                cas.cas_values[cas.commitment_layer],
                "o", color="red", markersize=12,
                label="Commitment", zorder=5,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("CAS")
        ax.set_title("Contextual Alignment Score across depth")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return fig

    def plot_layer_maps_grid(
        self,
        result: DSTResult,
        context_words: Optional[List[str]] = None,
        noncontext_words: Optional[List[str]] = None,
        cols: int = 3,
        figsize_per: Tuple[int, int] = (6, 5),
    ) -> plt.Figure:
        """
        Plot semantic maps at all map layers in a grid.
        """
        layers = sorted(result.semantic_maps.keys())
        n = len(layers)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(figsize_per[0] * cols, figsize_per[1] * rows),
        )
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        ctx_set = set(context_words) if context_words else set()
        nonctx_set = set(noncontext_words) if noncontext_words else set()

        for i, l in enumerate(layers):
            self.plot_semantic_map(
                result.semantic_maps[l],
                ax=axes_flat[i],
                node_color_ctx=ctx_set,
                node_color_nonctx=nonctx_set,
            )
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.tight_layout()
        return fig

    def plot_dst_summary(
        self,
        result: DSTResult,
        context_words: Optional[List[str]] = None,
        noncontext_words: Optional[List[str]] = None,
        map_layers: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (18, 10),
    ) -> plt.Figure:
        """
        Combined summary figure (resembling Figure 1 in the paper):
        left panel = CAS trace, right panels = semantic maps at selected layers.
        """
        if map_layers is None:
            map_layers = sorted(result.semantic_maps.keys())
        sel = [l for l in map_layers if l in result.semantic_maps]
        n_maps = max(len(sel), 1)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_maps, 2, width_ratios=[1, 1.5])

        # CAS panel (spans all rows in left column)
        ax_cas = fig.add_subplot(gs[:, 0])
        if result.cas_trace.cas_values:
            self.plot_cas_trace(result.cas_trace, ax=ax_cas)
        else:
            ax_cas.text(
                0.5, 0.5,
                "CAS not computed\n(provide context_words & noncontext_words)",
                ha="center", va="center", fontsize=10,
            )
            ax_cas.axis("off")

        # Semantic map panels (right column)
        ctx_set = set(context_words) if context_words else set()
        nonctx_set = set(noncontext_words) if noncontext_words else set()
        for i, l in enumerate(sel):
            ax_map = fig.add_subplot(gs[i, 1])
            self.plot_semantic_map(
                result.semantic_maps[l],
                ax=ax_map,
                node_color_ctx=ctx_set,
                node_color_nonctx=nonctx_set,
            )

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_next_token_probs(
        probs: Dict[str, float],
        top_n: int = 15,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (8, 4),
    ) -> plt.Figure:
        """Horizontal bar chart of next-token probabilities."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        sorted_items = sorted(
            probs.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        words = [w for w, _ in reversed(sorted_items)]
        vals = [v for _, v in reversed(sorted_items)]
        ax.barh(words, vals, color="#74b9ff")
        ax.set_xlabel("Probability")
        ax.set_title("Next-token distribution at answer position")
        return fig

    # ------------------------------------------------------------------
    # CAS vs hallucination scatter (Figure 3 in the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_cas_vs_hallucination(
        cas_values: List[float],
        hallucinated: List[bool],
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (7, 5),
    ) -> plt.Figure:
        """
        Scatter plot of final-layer CAS vs hallucination indicator.

        Parameters
        ----------
        cas_values  : list of float — final-layer CAS per example.
        hallucinated: list of bool  — True if the model hallucinated.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cas_arr = np.array(cas_values)
        hall_arr = np.array(hallucinated, dtype=float)

        ax.scatter(
            cas_arr, hall_arr,
            alpha=0.5, color="#e17055", edgecolors="white", s=40,
        )

        # Linear fit
        if len(cas_arr) > 2:
            z = np.polyfit(cas_arr, hall_arr, 1)
            p = np.poly1d(z)
            xs = np.linspace(cas_arr.min(), cas_arr.max(), 100)
            ax.plot(xs, p(xs), "--", color="#2d3436", linewidth=1.5)
            r = float(np.corrcoef(cas_arr, hall_arr)[0, 1])
            ax.set_title(f"CAS vs Hallucination Rate  (r = {r:.3f})")
        else:
            ax.set_title("CAS vs Hallucination Rate")

        ax.set_xlabel("Final-layer CAS")
        ax.set_ylabel("Hallucinated (1=yes)")
        ax.grid(True, alpha=0.3)
        return fig
