#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example client for Chronos-2 time series forecasting via vLLM pooling API.

This demonstrates how to use the extended pooling API to perform time series
forecasting with Chronos-2 models.

Usage:
    python examples/forecast_client_example.py
"""

import json

import requests


def forecast_univariate_basic(base_url: str = "http://localhost:8000"):
    """
    Example 1: Basic univariate time series forecasting.
    """
    print("\n=== Example 1: Basic Univariate Forecasting ===\n")

    payload = {
        "model": "amazon/chronos-2",
        "task": "forecast",
        "data": {
            "inputs": [{"target": [10.5, 11.2, 12.1, 11.8, 13.2, 14.0]}],
            "parameters": {"prediction_length": 3},
        },
    }

    response = requests.post(
        f"{base_url}/pooling",
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    print(f"Request:\n{json.dumps(payload, indent=2)}\n")
    print(f"Response Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

    return response.json()


def forecast_with_metadata(base_url: str = "http://localhost:8000"):
    """
    Example 2: Forecasting with timestamps and item IDs.
    """
    print("\n=== Example 2: Forecasting with Metadata ===\n")

    payload = {
        "model": "amazon/chronos-2",
        "task": "forecast",
        "data": {
            "inputs": [
                {
                    "target": [10.5, 11.2, 12.1, 11.8, 13.2],
                    "item_id": "product_A",
                    "start": "2024-01-01",
                }
            ],
            "parameters": {
                "prediction_length": 3,
                "freq": "D",
                "quantile_levels": [0.25, 0.5, 0.75],
            },
        },
    }

    response = requests.post(
        f"{base_url}/pooling",
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    print(f"Request:\n{json.dumps(payload, indent=2)}\n")
    print(f"Response Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

    return response.json()


def forecast_multivariate_with_covariates(base_url: str = "http://localhost:8000"):
    """
    Example 3: Multivariate forecasting with covariates.
    """
    print("\n=== Example 3: Multivariate with Covariates ===\n")

    payload = {
        "model": "amazon/chronos-2",
        "task": "forecast",
        "data": {
            "inputs": [
                {
                    "target": [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
                    "past_covariates": {
                        "temperature": [15.5, 16.0, 17.2, 18.1, 19.0],
                        "promotion": [0, 0, 1, 1, 0],
                    },
                    "future_covariates": {
                        "temperature": [20.0, 21.5, 22.0],
                        "promotion": [1, 1, 0],
                    },
                }
            ],
            "parameters": {"prediction_length": 3},
        },
    }

    response = requests.post(
        f"{base_url}/pooling",
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    print(f"Request:\n{json.dumps(payload, indent=2)}\n")
    print(f"Response Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

    return response.json()


def forecast_batch_with_cross_learning(base_url: str = "http://localhost:8000"):
    """
    Example 4: Batch forecasting with cross-learning enabled.
    """
    print("\n=== Example 4: Batch Forecasting with Cross-Learning ===\n")

    payload = {
        "model": "amazon/chronos-2",
        "task": "forecast",
        "data": {
            "inputs": [
                {"target": [10, 11, 12, 13, 14, 15], "item_id": "store_1"},
                {"target": [20, 21, 22, 23, 24, 25], "item_id": "store_2"},
                {"target": [15, 16, 17, 18, 19, 20], "item_id": "store_3"},
            ],
            "parameters": {
                "prediction_length": 7,
                "cross_learning": True,
                "batch_size": 100,
            },
        },
    }

    response = requests.post(
        f"{base_url}/pooling",
        headers={"Content-Type": "application/json"},
        json=payload,
    )

    print(f"Request:\n{json.dumps(payload, indent=2)}\n")
    print(f"Response Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")

    return response.json()


def main():
    """
    Run all examples.
    """
    base_url = "http://localhost:8000"

    print("=" * 70)
    print("Chronos-2 Time Series Forecasting Examples")
    print("=" * 70)

    try:
        # Example 1: Basic univariate forecasting
        forecast_univariate_basic(base_url)

        # Example 2: With metadata (timestamps and item IDs)
        forecast_with_metadata(base_url)

        # Example 3: Multivariate with covariates
        forecast_multivariate_with_covariates(base_url)

        # Example 4: Batch with cross-learning
        forecast_batch_with_cross_learning(base_url)

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to vLLM server at {base_url}")
        print("Please ensure the server is running with a Chronos-2 model.")
        print("\nTo start the server:")
        print("  vllm serve amazon/chronos-2-base --runner pooling")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
