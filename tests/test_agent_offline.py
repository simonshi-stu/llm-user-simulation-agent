"""Offline tests for the agent's pure logic.

These tests do not require websocietysimulator, an API key, or network
access. When the framework is missing, improved_agent_with_quality.py falls
back to local stubs, so this module stays importable.
"""

import logging

import pytest

from improved_agent_with_quality import (
    ImprovedSimulationAgent,
    ReviewQualityAnalyzer,
    UserProfileAnalyzer,
)


def make_review(stars, length, useful=0, funny=0, cool=0):
    return {
        "stars": stars,
        "text": "x" * length,
        "useful": useful,
        "funny": funny,
        "cool": cool,
    }


def parse(text):
    return ImprovedSimulationAgent.parse_review_result(None, text)


class TestUserProfileAnalyzer:
    def test_empty_history_returns_safe_defaults(self):
        profile = UserProfileAnalyzer.analyze_user_patterns([])
        assert profile["user_type"] == "新用户"
        assert profile["avg_stars"] == 3.0
        assert profile["review_count"] == 0
        assert profile["star_distribution"] == {}

    def test_positive_user_stats(self):
        reviews = [
            make_review(5, 160),
            make_review(4, 160),
            make_review(3, 20),
        ]
        profile = UserProfileAnalyzer.analyze_user_patterns(reviews)
        assert profile["avg_stars"] == 4.0
        assert profile["rating_tendency"] == "positive"
        assert profile["review_count"] == 3
        assert profile["star_distribution"] == {5: 1, 4: 1, 3: 1}
        assert profile["review_style"] == "concise"

    def test_critical_user(self):
        reviews = [make_review(2, 30), make_review(1, 30)]
        profile = UserProfileAnalyzer.analyze_user_patterns(reviews)
        assert profile["rating_tendency"] == "critical"

    @pytest.mark.parametrize(
        ("useful", "expected"),
        [(3.0, "high"), (1.0, "medium"), (0.0, "low")],
    )
    def test_useful_tendency_thresholds(self, useful, expected):
        reviews = [make_review(4, 60, useful=useful)]
        profile = UserProfileAnalyzer.analyze_user_patterns(reviews)
        assert profile["useful_tendency"] == expected

    def test_informative_engagement_style(self):
        reviews = [make_review(4, 200, useful=2.0)]
        profile = UserProfileAnalyzer.analyze_user_patterns(reviews)
        assert profile["engagement_style"] == "informative"

    def test_cool_tendency_is_profiled(self):
        profile = UserProfileAnalyzer.analyze_user_patterns(
            [make_review(4, 100, cool=2.0)]
        )

        assert profile["cool_tendency"] == "high"

    def test_format_contains_profile_fields(self):
        reviews = [make_review(5, 100, useful=2.0)]
        profile = UserProfileAnalyzer.analyze_user_patterns(reviews)
        text = UserProfileAnalyzer.format_user_analysis(profile)
        assert "用户特征分析" in text
        assert "信息价值倾向" in text
        assert "主要评论语言" in text

    def test_selects_recent_engaging_and_representative_reviews(self):
        reviews = [
            make_review(5, 80),
            make_review(2, 100, useful=10),
            make_review(4, 120),
            make_review(3, 90),
        ]
        selected = UserProfileAnalyzer.select_representative_reviews(reviews)

        assert len(selected) == 3
        assert selected[0] is reviews[0]
        assert any(review.get("useful") == 10 for review in selected)


class TestReviewQualityAnalyzer:
    def test_empty_reviews(self):
        result = ReviewQualityAnalyzer.analyze_review_qualities([])
        assert result["has_useful_examples"] is False
        assert result["has_funny_examples"] is False

    def test_selects_qualified_examples(self):
        reviews = [
            {"text": "a" * 60, "stars": 5, "useful": 3, "funny": 0},
            {"text": "b" * 60, "stars": 4, "useful": 0, "funny": 2},
            {"text": "c" * 60, "stars": 3, "useful": 1, "funny": 1},
            {"text": "d" * 60, "stars": 2, "useful": 0, "funny": 0},
        ]
        result = ReviewQualityAnalyzer.analyze_review_qualities(reviews)
        assert result["has_useful_examples"] is True
        assert result["has_funny_examples"] is True
        assert len(result["useful_reviews"]) == 1
        assert result["useful_reviews"][0]["stars"] == 5
        assert result["total_reviews"] == 4

    def test_examples_capped_at_two(self):
        reviews = [
            {"text": "a" * 60, "stars": 5, "useful": 3, "funny": 3}
            for _ in range(3)
        ]
        result = ReviewQualityAnalyzer.analyze_review_qualities(reviews)
        assert len(result["useful_reviews"]) == 2
        assert len(result["funny_reviews"]) == 2

    def test_selects_cool_examples(self):
        result = ReviewQualityAnalyzer.analyze_review_qualities(
            [{"text": "a" * 60, "stars": 4, "cool": 2}]
        )

        assert result["has_cool_examples"] is True
        assert result["cool_reviews"][0]["cool"] == 2


class TestReferenceSelection:
    def test_prefers_long_reviews_and_respects_top_k(self):
        reviews = [{"text": "x" * 60} for _ in range(4)]
        reviews.append({"text": "short"})
        selected = ImprovedSimulationAgent.get_relevant_reviews(
            None, reviews, top_k=3
        )
        assert len(selected) == 3
        assert all(len(r["text"]) > 50 for r in selected)

    def test_falls_back_when_too_few_long_reviews(self):
        reviews = [{"text": "x" * 60}, {"text": "x" * 60}, {"text": "short"}]
        selected = ImprovedSimulationAgent.get_relevant_reviews(
            None, reviews, top_k=5
        )
        assert len(selected) == 3

    def test_empty_input(self):
        assert ImprovedSimulationAgent.get_relevant_reviews(None, []) == []


class TestParseReviewResult:
    def test_standard_format(self):
        stars, review = parse("stars: 4.0\nreview: Nice place.")
        assert stars == 4.0
        assert review == "Nice place."

    def test_chinese_labels(self):
        stars, review = parse("星级: 5.0\n评论: 很好")
        assert stars == 5.0
        assert review == "很好"

    def test_missing_stars_falls_back_to_three(self):
        stars, _ = parse("review: just the text")
        assert stars == 3.0

    def test_missing_review_falls_back_to_default_text(self):
        _, review = parse("stars: 5")
        assert review == "不错的体验。"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("stars: 9\nreview: x", 5.0), ("stars: 0.4\nreview: x", 1.0)],
    )
    def test_out_of_range_stars_are_clamped(self, raw, expected):
        stars, _ = parse(raw)
        assert stars == expected

    def test_multiline_review_is_preserved(self):
        stars, review = parse(
            "stars: 4.0\nreview: First line.\nSecond line.\nThird line."
        )
        assert stars == 4.0
        assert review == "First line.\nSecond line.\nThird line."

    def test_review_body_may_start_on_the_next_line(self):
        _, review = parse("stars: 4.0\nreview:\nFirst line.\nSecond line.")
        assert review == "First line.\nSecond line."

    def test_review_body_keeps_colons(self):
        _, review = parse("stars: 4.0\nreview: Service: slow, food: great.")
        assert review == "Service: slow, food: great."

    def test_empty_review_body_falls_back_to_default_text(self):
        _, review = parse("stars: 4.0\nreview:")
        assert review == "不错的体验。"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("stars: 4.5", 5.0), ("stars: 2.5", 3.0), ("stars: 3.7", 4.0)],
    )
    def test_non_grid_stars_snap_half_up(self, raw, expected):
        stars, _ = parse(raw)
        assert stars == expected

    def test_off_grid_stars_are_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse("stars: 4.5\nreview: text")
        assert "4.5" in caplog.text
        assert "snapped" in caplog.text
