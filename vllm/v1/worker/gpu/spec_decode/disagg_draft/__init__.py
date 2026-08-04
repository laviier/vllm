# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Draft Speculation (based on SSD / Saguaro) integration for vLLM.

Disaggregated draft speculation disaggregates the draft model to a separate
GPU and pre-computes speculations for multiple verification outcomes in a
"speculation cache".
On cache hits (~88% at T=0), draft latency is fully hidden.

Reference: "Speculative Speculative Decoding" (arXiv:2603.03251v1)
   Tanishq Kumar, Tri Dao, Avner May — Stanford / Princeton / Together AI
"""
