# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class TimeSeriesInput(BaseModel):
    """Input time series data for forecasting."""

    target: list[float] | list[list[float]] = Field(
        ...,
        description="Historical time series values. "
        "1-D array for univariate, 2-D array for multivariate",
    )

    item_id: str | None = Field(
        default=None, description="Unique identifier for the time series"
    )

    start: str | None = Field(
        default=None,
        description="Start timestamp in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    )

    past_covariates: dict[str, list[float] | list[str]] | None = Field(
        default=None,
        description=(
            "Dictionary of past covariate arrays (numeric or categorical). "
            "Each array must match the length of the target"
        ),
    )

    future_covariates: dict[str, list[float] | list[str]] | None = Field(
        default=None,
        description="Dictionary of known future covariate arrays. "
        "Keys must be a subset of past_covariates. "
        "Each array must match prediction_length",
    )

    @field_validator("target")
    @classmethod
    def validate_target_length(
        cls, v: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        """Validate minimum target length and multivariate consistency."""
        if isinstance(v[0], list):
            # Multivariate: check first dimension length
            first_dim: list[float] = v[0]
            if len(first_dim) < 5:
                raise ValueError(
                    f"Target must contain at least 5 observations (received {len(first_dim)})"
                )
            # Validate all dimensions have same number of observations
            first_len = len(first_dim)
            for i, dim in enumerate(v[1:], start=1):
                dim_list: list[float] = dim  # type: ignore[assignment]
                if len(dim_list) != first_len:
                    raise ValueError(
                        f"All target dimensions must have same length. "
                        f"Dimension 0 has {first_len} observations, "
                        f"dimension {i} has {len(dim_list)}"
                    )
        else:
            # Univariate
            if len(v) < 5:
                raise ValueError(
                    f"Target must contain at least 5 observations (received {len(v)})"
                )
        return v

    @field_validator("start")
    @classmethod
    def validate_start_timestamp(cls, v: str | None) -> str | None:
        """Validate start timestamp is in valid ISO format."""
        if v is not None:
            try:
                # Try parsing as ISO 8601 datetime
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(
                    f"Invalid start timestamp format: {v}. "
                    f"Expected ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
                ) from e
        return v

    @model_validator(mode="after")
    def validate_covariates(self) -> "TimeSeriesInput":
        """Validate covariate lengths match target length."""
        # Get target length
        if isinstance(self.target[0], list):
            target_len = len(self.target[0])  # Multivariate
        else:
            target_len = len(self.target)  # Univariate

        # Validate past_covariates length
        if self.past_covariates is not None:
            for key, values in self.past_covariates.items():
                if len(values) != target_len:
                    raise ValueError(
                        f"Past covariate '{key}' length ({len(values)}) "
                        f"must match target length ({target_len})"
                    )

        # Note: future_covariates length validation happens in ForecastRequest
        # validator where we have access to prediction_length parameter

        return self


class ForecastParameters(BaseModel):
    """Parameters for time series forecasting."""

    prediction_length: int = Field(
        default=1, ge=1, le=1024, description="Number of future steps to forecast"
    )

    quantile_levels: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Quantile levels for uncertainty quantification. "
        "Each value must be between 0 and 1 (exclusive)",
    )

    freq: str | None = Field(
        default=None,
        description="Pandas frequency string (e.g., 'D' for daily, 'H' for hourly). "
        "Required if 'start' is provided in inputs",
    )

    batch_size: int = Field(default=256, ge=1, description="Batch size for inference")

    cross_learning: bool = Field(
        default=False,
        description="Enable information sharing across time series in batch",
    )

    @field_validator("quantile_levels")
    @classmethod
    def validate_quantiles(cls, v: list[float]) -> list[float]:
        """Validate quantile levels are in (0, 1) range."""
        for q in v:
            if not (0 < q < 1):
                raise ValueError(
                    f"Quantile levels must be between 0 and 1 (exclusive), got {q}"
                )
        return v


class ForecastRequest(OpenAIBaseModel):
    """Request format for time series forecasting via pooling API."""

    model: str = Field(..., description="Model name to use for forecasting")

    task: Literal["forecast"] = Field(
        default="forecast", description="Task type, must be 'forecast'"
    )

    data: dict[str, Any] = Field(
        ...,
        description=(
            "Forecast request data containing 'inputs' and optional 'parameters'"
        ),
    )

    @field_validator("data")
    @classmethod
    def validate_data_structure(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate the data field contains required structure."""
        if "inputs" not in v:
            raise ValueError("data must contain 'inputs' field")

        if not isinstance(v["inputs"], list):
            raise ValueError("data.inputs must be a list")

        if len(v["inputs"]) == 0:
            raise ValueError("data.inputs cannot be empty")

        if len(v["inputs"]) > 1024:
            raise ValueError(
                f"data.inputs may contain at most 1024 time series "
                f"(received {len(v['inputs'])})"
            )

        # Validate each input as TimeSeriesInput
        validated_inputs = []
        for i, ts_input in enumerate(v["inputs"]):
            try:
                validated_input = TimeSeriesInput(**ts_input)
                validated_inputs.append(validated_input)
            except Exception as e:
                raise ValueError(f"Invalid time series input at index {i}: {e}") from e

        # Validate parameters if present
        validated_params = None
        if "parameters" in v and v["parameters"] is not None:
            try:
                validated_params = ForecastParameters(**v["parameters"])
            except Exception as e:
                raise ValueError(f"Invalid parameters: {e}") from e

        # Cross-validate future_covariates length with prediction_length
        if validated_params is not None:
            prediction_length = validated_params.prediction_length
            for i, ts_input in enumerate(validated_inputs):
                if ts_input.future_covariates is not None:
                    for key, values in ts_input.future_covariates.items():
                        if len(values) != prediction_length:
                            raise ValueError(
                                f"Input {i}: future_covariate '{key}' length "
                                f"({len(values)}) must match prediction_length "
                                f"({prediction_length})"
                            )

        return v


class ForecastPrediction(BaseModel):
    """Single time series forecast result."""

    mean: list[float] | list[list[float]] = Field(
        ..., description="Point forecast (median). Shape matches input target"
    )

    item_id: str | None = Field(
        default=None, description="Echoed from input if provided"
    )

    start: str | None = Field(
        default=None, description="Start timestamp of forecast horizon"
    )

    class Config:
        extra = "allow"  # Allow dynamic quantile fields like "0.1", "0.5", "0.9"


class ForecastResponse(OpenAIBaseModel):
    """Response format for time series forecasting."""

    request_id: str = Field(..., description="Request identifier")

    created_at: int = Field(
        ..., description="Unix timestamp when the response was created"
    )

    data: dict[str, list[ForecastPrediction]] = Field(
        ..., description="Forecast results with 'predictions' key"
    )
