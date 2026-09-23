"""Tests for the structured JSON output contract and reflection policy."""

import logging

import pytest
from pydantic import ValidationError

from improved_agent_with_quality import (
    AgentOutput,
    ImprovedSimulationAgent,
    draft_needs_reflection,
    extract_json_payload,
)


def parse_with_status(text):
    return ImprovedSimulationAgent.parse_review_result_with_status(text)


class TestExtractJsonPayload:
    def test_plain_json(self):
        payload = extract_json_payload('{"stars": 4, "review": "ok"}')
        assert payload == {"stars": 4, "review": "ok"}

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n{"stars": 5, "review": "great"}\n```'
        assert extract_json_payload(text) == {"stars": 5, "review": "great"}

    def test_json_embedded_in_prose(self):
        text = 'Result: {"stars": 3, "review": "meh"} done.'
        assert extract_json_payload(text) == {"stars": 3, "review": "meh"}

    def test_missing_json_returns_none(self):
        assert extract_json_payload("stars: 4\nreview: ok") is None
        assert extract_json_payload("") is None


class TestParseStructuredOutput:
    def test_json_is_parsed_without_fallback(self):
        stars, review, used_fallback = parse_with_status(
            '{"stars": 4.0, "review": "Great coffee."}'
        )
        assert (stars, review, used_fallback) == (4.0, "Great coffee.", False)

    def test_json_off_grid_stars_are_snapped(self):
        stars, _, used_fallback = parse_with_status(
            '{"stars": 4.5, "review": "Nice."}'
        )
        assert stars == 5.0
        assert used_fallback is False

    def test_invalid_json_falls_back_to_labels(self, caplog):
        text = '{"stars": 4.0}\nstars: 2.0\nreview: legacy text'
        with caplog.at_level(logging.WARNING):
            stars, review, used_fallback = parse_with_status(text)
        assert (stars, review, used_fallback) == (2.0, "legacy text", False)

    def test_missing_review_in_json_falls_back_to_default(self):
        stars, review, used_fallback = parse_with_status('{"stars": 2.0}')
        assert stars == 3.0
        assert review == "不错的体验。"
        assert used_fallback is True

    def test_legacy_format_still_supported(self):
        stars, review, used_fallback = parse_with_status(
            "stars: 4.0\nreview: legacy ok"
        )
        assert (stars, review, used_fallback) == (4.0, "legacy ok", False)


class TestAgentOutput:
    def test_extra_fields_are_ignored(self):
        output = AgentOutput(stars=3, review="ok", extra="ignored")
        assert output.stars == 3.0

    def test_empty_review_is_rejected(self):
        with pytest.raises(ValidationError):
            AgentOutput(stars=3, review="   ")


class TestDraftNeedsReflection:
    def test_good_structured_draft_skips_reflection(self):
        draft = (
            '{"stars": 4.0, "review": "A long enough review about '
            'the coffee and the service."}'
        )
        assert draft_needs_reflection(draft) == (False, "ok")

    def test_short_structured_review_needs_reflection(self):
        draft = '{"stars": 4.0, "review": "ok"}'
        assert draft_needs_reflection(draft) == (True, "review_too_short")

    def test_legacy_draft_is_evaluated(self):
        needs, reason = draft_needs_reflection("stars: 4.0\nreview: short")
        assert needs is True
        assert reason == "review_too_short"

    def test_missing_stars_needs_reflection(self):
        assert draft_needs_reflection("review: some text") == (
            True,
            "missing_stars",
        )

    def test_off_grid_stars_needs_reflection(self):
        draft = "stars: 4.5\nreview: " + "long enough review text " * 3
        assert draft_needs_reflection(draft) == (True, "off_grid_stars")

    def test_empty_draft_needs_reflection(self):
        assert draft_needs_reflection("") == (True, "empty_draft")

    def test_generic_draft_needs_reflection(self):
        draft = (
            '{"stars": 5.0, "review": "Good place and good experience. '
            'Overall everything was good."}'
        )
        assert draft_needs_reflection(draft) == (True, "generic_review")

    def test_rating_text_mismatch_needs_reflection(self):
        draft = (
            '{"stars": 5.0, "review": "The service was terrible and the '
            'food was awful, so I would avoid this place."}'
        )
        assert draft_needs_reflection(draft) == (True, "rating_text_mismatch")
