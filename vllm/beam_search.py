# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    multi_modal_data: "MultiModalDataDict | None" = None
    mm_processor_kwargs: dict[str, Any] | None = None


@dataclass
class BeamState:
    """State for a single beam in beam search.

    This represents the state of one beam within a beam group,
    including its token sequence and cumulative log probability.
    """

    beam_id: int
    tokens: list[int]
    cum_logprob: float
    logprobs: list[dict[int, Logprob]] = field(default_factory=list)
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    text: str | None = None


@dataclass
class BeamGroup:
    """Group of beams for beam search optimization.

    This class groups multiple beams together to reduce scheduler overhead.
    Instead of tracking each beam as a separate request, all beams in a
    group share a single request ID and are managed together.

    Benefits:
    - Reduced scheduler overhead (1 dict entry instead of beam_width)
    - Batched KV cache operations
    - Better memory efficiency
    - Simplified request tracking

    Attributes:
        beam_width: Number of beams in this group
        beam_states: List of individual beam states
        shared_prefix_length: Length of the shared prefix across all beams
        iteration: Current iteration number in beam search
    """

    beam_width: int
    beam_states: list[BeamState]
    shared_prefix_length: int = 0
    iteration: int = 0

    def __post_init__(self):
        """Validate that beam_states matches beam_width."""
        if len(self.beam_states) != self.beam_width:
            raise ValueError(
                f"beam_states length ({len(self.beam_states)}) "
                f"does not match beam_width ({self.beam_width})"
            )

    def get_beam_by_id(self, beam_id: int) -> BeamState:
        """Get a specific beam by its ID."""
        if beam_id < 0 or beam_id >= self.beam_width:
            raise ValueError(f"beam_id {beam_id} out of range [0, {self.beam_width})")
        return self.beam_states[beam_id]

    def update_shared_prefix_length(self) -> None:
        """Calculate and update the shared prefix length across all beams."""
        if not self.beam_states:
            self.shared_prefix_length = 0
            return

        # Find the minimum length among all beams
        min_length = min(len(beam.tokens) for beam in self.beam_states)

        # Find the longest common prefix
        shared_length = 0
        first_beam_tokens = self.beam_states[0].tokens
        for i in range(min_length):
            if all(beam.tokens[i] == first_beam_tokens[i] for beam in self.beam_states):
                shared_length = i + 1
            else:
                break

        self.shared_prefix_length = shared_length

    def all_beams_finished(self) -> bool:
        """Check if all beams have finished."""
        return all(beam.finish_reason is not None for beam in self.beam_states)

    def get_active_beam_count(self) -> int:
        """Get the number of beams that haven't finished yet."""
        return sum(1 for beam in self.beam_states if beam.finish_reason is None)


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key
