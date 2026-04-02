#!/usr/bin/env python3
# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""System Usability Scale (SUS) scoring utility.

Reference: Brooke, J. (1996). "SUS: A quick and dirty usability scale."
Adjective ratings: Bangor, A., Kortum, P.T., Miller, J.T. (2008).
"""

from __future__ import annotations


def sus_score(responses: list[int]) -> float:
    """Compute the SUS score from 10 Likert-scale responses (1-5).

    Odd-numbered questions (1, 3, 5, 7, 9): subtract 1 from the response.
    Even-numbered questions (2, 4, 6, 8, 10): subtract the response from 5.
    Sum the adjusted scores and multiply by 2.5.

    Args:
        responses: List of 10 integers, each in [1, 5].

    Returns:
        SUS score in [0, 100].

    Raises:
        ValueError: If input is not exactly 10 responses in [1, 5].
    """
    if len(responses) != 10:
        raise ValueError(f"Expected 10 responses, got {len(responses)}")
    for i, r in enumerate(responses):
        if not (1 <= r <= 5):
            raise ValueError(f"Response {i + 1} must be 1-5, got {r}")

    adjusted = 0.0
    for i, r in enumerate(responses):
        if i % 2 == 0:
            adjusted += r - 1
        else:
            adjusted += 5 - r
    return adjusted * 2.5


def adjective_rating(score: float) -> str:
    """Map a SUS score to an adjective rating (Bangor et al. 2008).

    Returns:
        One of: "Excellent", "Good", "OK", "Poor", "Awful".
    """
    if score >= 85.5:
        return "Excellent"
    if score >= 71.4:
        return "Good"
    if score >= 50.9:
        return "OK"
    if score >= 35.7:
        return "Poor"
    return "Awful"


def main() -> None:
    """Interactive CLI for quick SUS calculation."""
    print("Enter 10 SUS responses (1-5), space-separated:")
    raw = input("> ").strip().split()
    responses = [int(x) for x in raw]
    score = sus_score(responses)
    rating = adjective_rating(score)
    print(f"\nSUS Score: {score:.1f} / 100")
    print(f"Rating:    {rating}")


if __name__ == "__main__":
    main()
