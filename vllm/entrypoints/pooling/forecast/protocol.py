# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
        """Validate minimum target length."""
        if isinstance(v[0], list):
            # Multivariate: check first dimension length
            if len(v[0]) < 5:
                raise ValueError(
                    f"Target must contain at least 5 observations(received {len(v[0])})"
                )
        else:
            # Univariate
            if len(v) < 5:
                raise ValueError(
                    f"Target must contain at least 5 observations(received {len(v)})"
                )
        return v


class ForecastParameters(BaseModel):
    """Parameters for time series forecasting."""

    prediction_length: int = Field(
        default=1, ge=1, le=1024, description="Number of future steps to forecast"
    )

    quantile_levels: list[float] | None = Field(
        default=None,
        description="Quantile levels for uncertainty quantification. "
        "Each value must be between 0 and 1 (exclusive). "
        "Default: [0.1, 0.5, 0.9]",
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
    def validate_quantiles(cls, v: list[float] | None) -> list[float] | None:
        """Validate quantile levels are in (0, 1) range."""
        if v is not None:
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
        for i, ts_input in enumerate(v["inputs"]):
            try:
                TimeSeriesInput(**ts_input)
            except Exception as e:
                raise ValueError(f"Invalid time series input at index {i}: {e}") from e

        # Validate parameters if present
        if "parameters" in v and v["parameters"] is not None:
            try:
                ForecastParameters(**v["parameters"])
            except Exception as e:
                raise ValueError(f"Invalid parameters: {e}") from e

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
