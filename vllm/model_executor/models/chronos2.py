# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chronos-2: Encoder-only time series foundation model for zero-shot forecasting."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import VllmModelForPooling

logger = init_logger(__name__)


class Chronos2Pooler(Pooler):
    """Simple pooler for Chronos-2 time series model."""

    def get_supported_tasks(self) -> set[PoolingTask]:
        """Return supported pooling tasks."""
        return {"forecast"}  # Chronos-2 is a time series forecasting model

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        """
        Pool hidden states for Chronos-2.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            pooling_metadata: Pooling metadata

        Returns:
            Pooled tensor (batch, hidden_size)
        """
        # Mean pooling across sequence dimension
        return hidden_states.mean(dim=1)


class Chronos2ForForecasting(nn.Module, VllmModelForPooling):
    """
    Chronos-2 time series forecasting model.

    This is a 120M-parameter, encoder-only foundation model for zero-shot
    time series forecasting. It integrates with the chronos-forecasting package
    for actual inference.
    """

    packed_modules_mapping = {}
    supported_lora_modules = []
    embedding_modules = {}
    embedding_padding_modules = []

    is_pooling_model = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # CRITICAL: Chronos-2 is encoder-only for time series forecasting,
        # not an encoder-decoder model. Override the T5-based config.
        config.is_encoder_decoder = False

        self.config = config
        self.model_name = vllm_config.model_config.model

        # Store model configuration
        self.d_model = getattr(config, "d_model", 768)
        self.num_layers = getattr(config, "num_layers", 12)
        self.num_heads = getattr(config, "num_heads", 12)

        # Chronos-2 specific config
        chronos_config = getattr(config, "chronos_config", {})
        self.context_length = chronos_config.get("context_length", 8192)
        self.input_patch_size = chronos_config.get("input_patch_size", 16)
        self.output_patch_size = chronos_config.get("output_patch_size", 16)
        self.max_output_patches = chronos_config.get("max_output_patches", 64)
        self.quantiles = chronos_config.get(
            "quantiles",
            [
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.99,
            ],
        )

        # Initialize Chronos2Pipeline (lazy loading)
        self._pipeline = None

        # Initialize pooler module
        self.pooler = Chronos2Pooler()

        logger.info(
            "Initialized Chronos2ForForecasting with d_model=%d, "
            "num_layers=%d, context_length=%d",
            self.d_model,
            self.num_layers,
            self.context_length,
        )

    @property
    def pipeline(self):
        """Lazy load the Chronos2Pipeline."""
        if self._pipeline is None:
            try:
                from chronos import Chronos2Pipeline

                logger.info("Loading Chronos2Pipeline for model: %s", self.model_name)
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.model_name,
                    device_map="cuda",
                )
            except ImportError as e:
                logger.error(
                    "Failed to import chronos-forecasting. "
                    "Install with: pip install chronos-forecasting"
                )
                raise ImportError(
                    "chronos-forecasting not installed. "
                    "Install with: pip install chronos-forecasting"
                ) from e
        return self._pipeline

    def predict(
        self,
        inputs: list[dict[str, Any]],
        prediction_length: int = 1,
        batch_size: int = 256,
        cross_learning: bool = False,
        quantile_levels: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate forecasts for time series inputs.

        Args:
            inputs: List of time series dictionaries with 'target' and optional covariates
            prediction_length: Number of future timesteps to forecast
            batch_size: Batch size for inference
            cross_learning: Whether to enable cross-series learning
            quantile_levels: List of quantile levels to return (defaults to [0.1, 0.5, 0.9])

        Returns:
            List of prediction dictionaries with quantile forecasts
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        # Convert inputs to Chronos2 format
        chronos_inputs = []
        for ts in inputs:
            input_dict = {
                "target": (
                    torch.tensor(ts["target"], dtype=torch.float32)
                    if not isinstance(ts["target"], torch.Tensor)
                    else ts["target"]
                ),
            }

            # Add past covariates if present
            if ts.get("past_covariates"):
                past_cov = {}
                for k, v in ts["past_covariates"].items():
                    if not isinstance(v, torch.Tensor):
                        past_cov[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        past_cov[k] = v
                input_dict["past_covariates"] = past_cov

            # Add future covariates if present
            if ts.get("future_covariates"):
                future_cov = {}
                for k, v in ts["future_covariates"].items():
                    if not isinstance(v, torch.Tensor):
                        future_cov[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        future_cov[k] = v
                input_dict["future_covariates"] = future_cov

            chronos_inputs.append(input_dict)

        # Run inference using Chronos2Pipeline
        forecast_results = self.pipeline.predict(
            inputs=chronos_inputs,
            prediction_length=prediction_length,
            batch_size=batch_size,
            cross_learning=cross_learning,
            limit_prediction_length=False,
        )

        # Format predictions
        predictions = []
        for idx, forecast_samples in enumerate(forecast_results):
            # forecast_samples shape: (num_samples, prediction_length)
            # We need to compute quantiles across the sample dimension

            pred = {}

            # Convert quantile levels to tensor for computation
            quantile_tensor = torch.tensor(
                quantile_levels,
                dtype=forecast_samples.dtype,
                device=forecast_samples.device,
            )

            # Compute quantiles across sample dimension (dim=0)
            # forecast_quantiles shape: (num_quantiles, prediction_length)
            forecast_quantiles = torch.quantile(
                forecast_samples, quantile_tensor, dim=0
            )

            # Extract each requested quantile
            for q_idx, q in enumerate(quantile_levels):
                pred[str(q)] = forecast_quantiles[q_idx].cpu().numpy().tolist()

            # Add mean (average across samples)
            pred["mean"] = forecast_samples.mean(dim=0).cpu().numpy().tolist()

            # Add metadata if present in input
            if "item_id" in inputs[idx]:
                pred["item_id"] = inputs[idx]["item_id"]
            if "start" in inputs[idx]:
                pred["start"] = inputs[idx]["start"]

            predictions.append(pred)

        return predictions

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for Chronos-2 model.

        This is a placeholder for vLLM compatibility during model initialization.
        Actual forecasting uses the `predict` method with Chronos2Pipeline directly.

        Args:
            input_ids: Time series input tensor
            positions: Position indices
            intermediate_tensors: Optional intermediate tensors
            inputs_embeds: Optional pre-computed embeddings

        Returns:
            Dummy embeddings tensor for vLLM compatibility
        """
        # For vLLM kernel warmup and initialization compatibility
        # Return dummy output since actual forecasting uses Chronos2Pipeline

        if inputs_embeds is not None:
            return inputs_embeds

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Handle both 1D (warmup) and 2D (normal) input shapes
        if input_ids.ndim == 1:
            # During warmup, vLLM may pass 1D input
            batch_size = 1
            seq_len = input_ids.shape[0]
        elif input_ids.ndim == 2:
            batch_size, seq_len = input_ids.shape
        else:
            raise ValueError(f"Expected 1D or 2D input_ids, got {input_ids.ndim}D")

        # Return dummy embeddings for vLLM compatibility
        # Actual forecasting happens in predict() method with Chronos2Pipeline
        embeddings = torch.zeros(
            batch_size,
            seq_len,
            self.d_model,
            device=input_ids.device,
            dtype=torch.float32,
        )
        return embeddings

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input IDs to embeddings.

        For Chronos-2, this is a placeholder since actual embedding
        happens in the Chronos2Pipeline.

        Args:
            input_ids: Input tensor

        Returns:
            Dummy embeddings
        """
        # Handle both 1D and 2D inputs
        if input_ids.ndim == 1:
            batch_size = 1
            seq_len = input_ids.shape[0]
        else:
            batch_size, seq_len = input_ids.shape

        return torch.zeros(
            batch_size,
            seq_len,
            self.d_model,
            device=input_ids.device,
            dtype=torch.float32,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for Chronos-2 model.

        Note: Weights are primarily managed by the Chronos2Pipeline from the
        chronos-forecasting package. This method is provided for compatibility
        with vLLM's weight loading infrastructure.

        Returns:
            Set of loaded parameter names (empty since weights handled externally)
        """
        logger.info(
            "Chronos-2 weights are managed by chronos-forecasting package. "
            "Model will be loaded via Chronos2Pipeline.from_pretrained()"
        )
        # The actual model weights are loaded when the pipeline property is accessed
        return set()  # Return empty set since weights handled by Chronos2Pipeline
