"""Tests for the FACS emotion analysis module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from raphael.emotion_analysis import (
    EMOTION_AU_MAP,
    _map_aus_to_emotions,
    analyze_emotion,
    detect_action_units,
)


# ---------------------------------------------------------------------------
# Tests for analyze_emotion – invalid inputs
# ---------------------------------------------------------------------------


class TestAnalyzeEmotionInvalidInputs:
    """Verify that analyze_emotion handles invalid inputs gracefully."""

    def test_none_input(self):
        result = analyze_emotion(None)
        assert result["face_detected"] is False
        assert result["dominant_emotion"] == "neutral"
        assert result["action_units"] == {}
        assert result["emotions"] == {}

    def test_empty_array(self):
        result = analyze_emotion(np.array([]))
        assert result["face_detected"] is False
        assert result["dominant_emotion"] == "neutral"

    def test_non_numpy_input(self):
        result = analyze_emotion("not an image")
        assert result["face_detected"] is False
        assert result["dominant_emotion"] == "neutral"


# ---------------------------------------------------------------------------
# Tests for analyze_emotion – no face detected
# ---------------------------------------------------------------------------


class TestAnalyzeEmotionNoFace:
    """Verify behaviour when no face is present in the image."""

    def test_blank_image(self):
        """A blank image should not contain a face."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyze_emotion(blank)
        assert result["face_detected"] is False
        assert result["dominant_emotion"] == "neutral"
        assert result["action_units"] == {}
        assert result["emotions"] == {}


# ---------------------------------------------------------------------------
# Tests for detect_action_units – invalid inputs
# ---------------------------------------------------------------------------


class TestDetectActionUnitsInvalidInputs:
    """Verify that detect_action_units handles invalid inputs gracefully."""

    def test_none_input(self):
        assert detect_action_units(None) == []

    def test_empty_array(self):
        assert detect_action_units(np.array([])) == []

    def test_blank_image_no_face(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert detect_action_units(blank) == []


# ---------------------------------------------------------------------------
# Tests for the AU → emotion mapping logic
# ---------------------------------------------------------------------------


class TestMapAUsToEmotions:
    """Test the internal _map_aus_to_emotions function."""

    def test_no_active_aus(self):
        """All AU values below threshold → no emotions detected."""
        aus = {au: 0.0 for au in ["AU1", "AU2", "AU4", "AU5", "AU6", "AU12"]}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert emotions == {}

    def test_happiness_detected(self):
        """AU6 + AU12 above threshold → happiness."""
        aus = {"AU6": 0.8, "AU12": 0.9, "AU25": 0.5}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "happiness" in emotions
        assert emotions["happiness"] > 0

    def test_sadness_detected(self):
        """AU1 + AU4 above threshold → sadness."""
        aus = {"AU1": 0.7, "AU4": 0.6, "AU15": 0.5}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "sadness" in emotions

    def test_surprise_detected(self):
        """AU1 + AU2 + AU5 + AU27 above threshold → surprise."""
        aus = {"AU1": 0.8, "AU2": 0.7, "AU5": 0.9, "AU27": 0.6}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "surprise" in emotions

    def test_anger_detected(self):
        """AU4 + AU5 above threshold → anger."""
        aus = {"AU4": 0.7, "AU5": 0.6, "AU7": 0.5, "AU23": 0.4}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "anger" in emotions

    def test_disgust_detected(self):
        """AU9 above threshold → disgust."""
        aus = {"AU9": 0.8, "AU15": 0.4}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "disgust" in emotions

    def test_contempt_detected(self):
        """AU12R + AU14R above threshold → contempt."""
        aus = {"AU12R": 0.6, "AU14R": 0.5}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "contempt" in emotions

    def test_fear_detected(self):
        """AU1 + AU2 + AU4 above threshold → fear."""
        aus = {"AU1": 0.6, "AU2": 0.5, "AU4": 0.7}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "fear" in emotions

    def test_required_below_threshold(self):
        """If a required AU is below threshold, emotion is not detected."""
        aus = {"AU6": 0.8, "AU12": 0.1}  # AU12 below 0.3
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        assert "happiness" not in emotions

    def test_optional_boost(self):
        """Optional AUs should increase confidence score."""
        base = {"AU6": 0.8, "AU12": 0.8}
        with_optional = {"AU6": 0.8, "AU12": 0.8, "AU25": 0.6}
        score_base = _map_aus_to_emotions(base, threshold=0.3).get("happiness", 0)
        score_opt = _map_aus_to_emotions(with_optional, threshold=0.3).get(
            "happiness", 0
        )
        assert score_opt >= score_base

    def test_confidence_capped_at_one(self):
        """Confidence scores should never exceed 1.0."""
        aus = {"AU6": 1.0, "AU12": 1.0, "AU25": 1.0}
        emotions = _map_aus_to_emotions(aus, threshold=0.3)
        for score in emotions.values():
            assert score <= 1.0

    def test_empty_aus(self):
        """Empty AU dict → no emotions."""
        assert _map_aus_to_emotions({}) == {}


# ---------------------------------------------------------------------------
# Tests for analyze_emotion with mocked face detection
# ---------------------------------------------------------------------------


class TestAnalyzeEmotionWithMock:
    """Test analyze_emotion by mocking the face landmark detection."""

    @staticmethod
    def _make_landmark(x, y):
        lm = MagicMock()
        lm.x = x
        lm.y = y
        return lm

    def _make_neutral_landmarks(self, n=478):
        """Create a set of landmarks arranged in a neutral expression."""
        landmarks = [self._make_landmark(0.5, 0.5) for _ in range(n)]

        # Place reference points for a realistic face layout
        # Forehead (idx 10) at top, Chin (idx 152) at bottom
        landmarks[10] = self._make_landmark(0.5, 0.1)  # forehead
        landmarks[152] = self._make_landmark(0.5, 0.9)  # chin

        # Eyebrows
        landmarks[107] = self._make_landmark(0.42, 0.25)  # left inner brow
        landmarks[70] = self._make_landmark(0.30, 0.24)  # left outer brow
        landmarks[336] = self._make_landmark(0.58, 0.25)  # right inner brow
        landmarks[300] = self._make_landmark(0.70, 0.24)  # right outer brow

        # Eyes
        landmarks[159] = self._make_landmark(0.37, 0.30)  # left eye upper
        landmarks[145] = self._make_landmark(0.37, 0.33)  # left eye lower
        landmarks[133] = self._make_landmark(0.40, 0.32)  # left eye inner
        landmarks[33] = self._make_landmark(0.32, 0.32)  # left eye outer
        landmarks[386] = self._make_landmark(0.63, 0.30)  # right eye upper
        landmarks[374] = self._make_landmark(0.63, 0.33)  # right eye lower
        landmarks[362] = self._make_landmark(0.60, 0.32)  # right eye inner
        landmarks[263] = self._make_landmark(0.68, 0.32)  # right eye outer

        # Nose
        landmarks[1] = self._make_landmark(0.50, 0.50)  # nose tip
        landmarks[6] = self._make_landmark(0.50, 0.35)  # nose bridge

        # Mouth
        landmarks[13] = self._make_landmark(0.50, 0.65)  # upper lip top
        landmarks[14] = self._make_landmark(0.50, 0.70)  # lower lip bottom
        landmarks[61] = self._make_landmark(0.40, 0.67)  # mouth left
        landmarks[291] = self._make_landmark(0.60, 0.67)  # mouth right
        landmarks[0] = self._make_landmark(0.50, 0.67)  # upper lip inner
        landmarks[17] = self._make_landmark(0.50, 0.68)  # lower lip inner

        return landmarks

    @patch("raphael.emotion_analysis._detect_face_landmarks")
    def test_face_detected_returns_correct_structure(self, mock_detect):
        """When a face is detected the result dict has the expected keys."""
        face = MagicMock()
        face.landmark = self._make_neutral_landmarks()
        mock_detect.return_value = [face]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyze_emotion(frame)

        assert result["face_detected"] is True
        assert isinstance(result["action_units"], dict)
        assert isinstance(result["emotions"], dict)
        assert isinstance(result["dominant_emotion"], str)

    @patch("raphael.emotion_analysis._detect_face_landmarks")
    def test_action_units_have_valid_range(self, mock_detect):
        """All AU intensities should be between 0 and 1."""
        face = MagicMock()
        face.landmark = self._make_neutral_landmarks()
        mock_detect.return_value = [face]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyze_emotion(frame)

        for au_name, intensity in result["action_units"].items():
            assert 0.0 <= intensity <= 1.0, f"{au_name} out of range: {intensity}"

    @patch("raphael.emotion_analysis._detect_face_landmarks")
    def test_no_face_returns_neutral(self, mock_detect):
        """When no face is detected, dominant emotion is 'neutral'."""
        mock_detect.return_value = []

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyze_emotion(frame)

        assert result["face_detected"] is False
        assert result["dominant_emotion"] == "neutral"


# ---------------------------------------------------------------------------
# Tests for EMOTION_AU_MAP structure
# ---------------------------------------------------------------------------


class TestEmotionAUMap:
    """Validate the EMOTION_AU_MAP configuration."""

    def test_all_emotions_have_required_key(self):
        for emotion, cfg in EMOTION_AU_MAP.items():
            assert "required" in cfg, f"{emotion} missing 'required' key"
            assert "optional" in cfg, f"{emotion} missing 'optional' key"

    def test_all_emotions_have_at_least_one_required_au(self):
        for emotion, cfg in EMOTION_AU_MAP.items():
            assert len(cfg["required"]) > 0, f"{emotion} has no required AUs"

    def test_seven_basic_emotions(self):
        expected = {
            "happiness",
            "sadness",
            "surprise",
            "fear",
            "anger",
            "disgust",
            "contempt",
        }
        assert set(EMOTION_AU_MAP.keys()) == expected
