"""
Module d'analyse des émotions basé sur le système FACS (Facial Action Coding System).

Utilise MediaPipe Face Mesh pour la détection des landmarks faciaux,
calcule les Action Units (AU) du FACS, et détermine les émotions
à partir des combinaisons d'AUs détectées.
"""

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# ---------------------------------------------------------------------------

# Eyebrows
_LEFT_INNER_BROW = 107
_LEFT_OUTER_BROW = 70
_RIGHT_INNER_BROW = 336
_RIGHT_OUTER_BROW = 300

# Eyes
_LEFT_EYE_UPPER = 159
_LEFT_EYE_LOWER = 145
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_UPPER = 386
_RIGHT_EYE_LOWER = 374
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263

# Nose
_NOSE_TIP = 1
_NOSE_BRIDGE = 6

# Mouth
_UPPER_LIP_TOP = 13
_LOWER_LIP_BOTTOM = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_UPPER_LIP_INNER = 0
_LOWER_LIP_INNER = 17

# Face reference points
_FOREHEAD = 10
_CHIN = 152

# ---------------------------------------------------------------------------
# FACS Emotion mappings (AU combinations → emotions)
# Based on Ekman & Friesen's FACS coding manual
# ---------------------------------------------------------------------------

EMOTION_AU_MAP = {
    "happiness": {"required": ["AU6", "AU12"], "optional": ["AU25"]},
    "sadness": {"required": ["AU1", "AU4"], "optional": ["AU15", "AU17"]},
    "surprise": {"required": ["AU1", "AU2", "AU5", "AU27"], "optional": ["AU26"]},
    "fear": {"required": ["AU1", "AU2", "AU4"], "optional": ["AU5", "AU20", "AU26"]},
    "anger": {"required": ["AU4", "AU5"], "optional": ["AU7", "AU23", "AU24"]},
    "disgust": {"required": ["AU9"], "optional": ["AU15", "AU25"]},
    "contempt": {"required": ["AU12R", "AU14R"], "optional": []},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_landmark_point(landmarks, index, w, h):
    """Extract (x, y) pixel coordinates from a MediaPipe landmark."""
    lm = landmarks[index]
    return np.array([lm.x * w, lm.y * h])


def _compute_distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def _detect_face_landmarks(frame):
    """
    Detect face landmarks using MediaPipe Face Mesh.

    Returns:
        list: List of face landmark sets (one per detected face).
    """
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks
    return []


# ---------------------------------------------------------------------------
# Action Unit computation
# ---------------------------------------------------------------------------


def _compute_action_units(landmarks, w, h):
    """
    Compute FACS Action Unit intensities from face landmarks.

    Uses geometric relationships between facial landmarks, normalized by
    face height, to estimate each AU's activation intensity on a 0–1 scale.
    """

    def pt(idx):
        return _get_landmark_point(landmarks, idx, w, h)

    # Reference distance for normalization (face height)
    face_height = _compute_distance(pt(_FOREHEAD), pt(_CHIN))
    if face_height < 1e-6:
        return {}

    aus = {}

    # --- AU1: Inner Brow Raiser ---
    left_inner_brow_dist = _compute_distance(pt(_LEFT_INNER_BROW), pt(_LEFT_EYE_INNER))
    right_inner_brow_dist = _compute_distance(
        pt(_RIGHT_INNER_BROW), pt(_RIGHT_EYE_INNER)
    )
    au1 = ((left_inner_brow_dist + right_inner_brow_dist) / 2) / face_height
    aus["AU1"] = min(max((au1 - 0.06) / 0.04, 0.0), 1.0)

    # --- AU2: Outer Brow Raiser ---
    left_outer_brow_dist = _compute_distance(pt(_LEFT_OUTER_BROW), pt(_LEFT_EYE_OUTER))
    right_outer_brow_dist = _compute_distance(
        pt(_RIGHT_OUTER_BROW), pt(_RIGHT_EYE_OUTER)
    )
    au2 = ((left_outer_brow_dist + right_outer_brow_dist) / 2) / face_height
    aus["AU2"] = min(max((au2 - 0.06) / 0.04, 0.0), 1.0)

    # --- AU4: Brow Lowerer ---
    brow_center_dist = (
        _compute_distance(pt(_LEFT_INNER_BROW), pt(_NOSE_BRIDGE))
        + _compute_distance(pt(_RIGHT_INNER_BROW), pt(_NOSE_BRIDGE))
    ) / 2
    au4_raw = brow_center_dist / face_height
    aus["AU4"] = min(max((0.12 - au4_raw) / 0.04, 0.0), 1.0)

    # --- AU5: Upper Lid Raiser ---
    left_eye_opening = _compute_distance(pt(_LEFT_EYE_UPPER), pt(_LEFT_EYE_LOWER))
    right_eye_opening = _compute_distance(pt(_RIGHT_EYE_UPPER), pt(_RIGHT_EYE_LOWER))
    au5_raw = ((left_eye_opening + right_eye_opening) / 2) / face_height
    aus["AU5"] = min(max((au5_raw - 0.04) / 0.03, 0.0), 1.0)

    # --- AU6: Cheek Raiser ---
    left_cheek = _compute_distance(pt(_LEFT_EYE_LOWER), pt(_MOUTH_LEFT))
    right_cheek = _compute_distance(pt(_RIGHT_EYE_LOWER), pt(_MOUTH_RIGHT))
    au6_raw = ((left_cheek + right_cheek) / 2) / face_height
    aus["AU6"] = min(max((0.28 - au6_raw) / 0.06, 0.0), 1.0)

    # --- AU7: Lid Tightener ---
    au7_raw = ((left_eye_opening + right_eye_opening) / 2) / face_height
    aus["AU7"] = min(max((0.04 - au7_raw) / 0.02, 0.0), 1.0)

    # --- AU9: Nose Wrinkler ---
    nose_to_lip = _compute_distance(pt(_NOSE_TIP), pt(_UPPER_LIP_TOP))
    au9_raw = nose_to_lip / face_height
    aus["AU9"] = min(max((0.10 - au9_raw) / 0.04, 0.0), 1.0)

    # --- AU12: Lip Corner Puller (smile) ---
    mouth_width = _compute_distance(pt(_MOUTH_LEFT), pt(_MOUTH_RIGHT))
    mouth_center_y = (pt(_MOUTH_LEFT)[1] + pt(_MOUTH_RIGHT)[1]) / 2
    lip_top_y = pt(_UPPER_LIP_TOP)[1]
    au12_raw = mouth_width / face_height
    aus["AU12"] = min(max((au12_raw - 0.25) / 0.10, 0.0), 1.0)

    # --- AU15: Lip Corner Depressor ---
    corner_depress = (mouth_center_y - lip_top_y) / face_height
    aus["AU15"] = min(max(corner_depress / 0.02, 0.0), 1.0)

    # --- AU17: Chin Raiser ---
    chin_to_lip = _compute_distance(pt(_CHIN), pt(_LOWER_LIP_BOTTOM))
    au17_raw = chin_to_lip / face_height
    aus["AU17"] = min(max((0.15 - au17_raw) / 0.05, 0.0), 1.0)

    # --- AU20: Lip Stretcher ---
    aus["AU20"] = min(max((au12_raw - 0.28) / 0.08, 0.0), 1.0)

    # --- AU23: Lip Tightener ---
    lip_thickness = _compute_distance(pt(_UPPER_LIP_TOP), pt(_LOWER_LIP_BOTTOM))
    au23_raw = lip_thickness / face_height
    aus["AU23"] = min(max((0.06 - au23_raw) / 0.03, 0.0), 1.0)

    # --- AU24: Lip Pressor ---
    # AU24 co-activates with AU23 (both involve orbicularis oris muscle)
    # but at lower intensity since pressing requires less effort than tightening.
    aus["AU24"] = aus["AU23"] * 0.8

    # --- AU25: Lips Part ---
    inner_lip_dist = _compute_distance(pt(_UPPER_LIP_INNER), pt(_LOWER_LIP_INNER))
    au25_raw = inner_lip_dist / face_height
    aus["AU25"] = min(max(au25_raw / 0.03, 0.0), 1.0)

    # --- AU26: Jaw Drop ---
    jaw_opening = _compute_distance(pt(_UPPER_LIP_TOP), pt(_CHIN))
    au26_raw = jaw_opening / face_height
    aus["AU26"] = min(max((au26_raw - 0.30) / 0.08, 0.0), 1.0)

    # --- AU27: Mouth Stretch ---
    mouth_opening = _compute_distance(pt(_UPPER_LIP_INNER), pt(_LOWER_LIP_INNER))
    au27_raw = mouth_opening / face_height
    aus["AU27"] = min(max((au27_raw - 0.04) / 0.06, 0.0), 1.0)

    # --- Asymmetric AUs for contempt ---
    left_corner_y = pt(_MOUTH_LEFT)[1]
    right_corner_y = pt(_MOUTH_RIGHT)[1]
    asymmetry = abs(left_corner_y - right_corner_y) / face_height
    aus["AU12R"] = min(max(asymmetry / 0.02, 0.0), 1.0)
    # AU14R (Dimpler) co-occurs with AU12R at reduced intensity because the
    # buccinator activation in dimpling is secondary to the zygomatic pull.
    aus["AU14R"] = aus["AU12R"] * 0.7

    return aus


# ---------------------------------------------------------------------------
# Emotion mapping
# ---------------------------------------------------------------------------


def _map_aus_to_emotions(aus, threshold=0.3):
    """
    Map detected Action Units to emotions using FACS emotion prototypes.

    Args:
        aus: dict mapping AU name to intensity (0–1).
        threshold: minimum AU intensity to consider it active.

    Returns:
        dict: mapping emotion name to confidence score (0–1).
    """
    emotions = {}

    for emotion, au_config in EMOTION_AU_MAP.items():
        required = au_config["required"]
        optional = au_config["optional"]

        required_scores = [aus.get(au, 0.0) for au in required]

        if not required_scores:
            continue

        # All required AUs must be above threshold
        if all(s >= threshold for s in required_scores):
            required_avg = sum(required_scores) / len(required_scores)

            # Optional AUs boost confidence
            optional_scores = [aus.get(au, 0.0) for au in optional]
            optional_bonus = sum(s for s in optional_scores if s >= threshold)
            optional_bonus = optional_bonus / max(len(optional), 1) * 0.2

            confidence = min(required_avg + optional_bonus, 1.0)
            emotions[emotion] = round(confidence, 3)

    return emotions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_action_units(frame):
    """
    Détecte les Action Units FACS à partir d'une image de visage.

    Utilise la géométrie des landmarks faciaux pour estimer l'activation
    des différentes Action Units.

    Args:
        frame: numpy array (image BGR, tel que lu par OpenCV).

    Returns:
        list[dict]: Liste de dictionnaires (un par visage détecté).
            Chaque dictionnaire associe le nom de l'AU (str) à son
            intensité d'activation (float entre 0 et 1).
            Retourne une liste vide si aucun visage n'est détecté.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    face_landmarks_list = _detect_face_landmarks(frame)
    if not face_landmarks_list:
        return []

    h, w = frame.shape[:2]
    results = []

    for face_landmarks in face_landmarks_list:
        lm = face_landmarks.landmark
        aus = _compute_action_units(lm, w, h)
        results.append(aus)

    return results


def analyze_emotion(frame):
    """
    Analyse les émotions détectées dans une image en utilisant la méthode FACS.

    Détecte les visages, calcule les Action Units (AU) du système FACS
    (Facial Action Coding System), et détermine les émotions correspondantes.

    Args:
        frame: Image numpy (format BGR, tel que lu par OpenCV).

    Returns:
        dict: Dictionnaire contenant:
            - "face_detected" (bool): Si un visage a été détecté.
            - "action_units" (dict): AUs détectées avec leurs intensités (0–1).
            - "emotions" (dict): Émotions détectées avec leurs scores de confiance.
            - "dominant_emotion" (str): L'émotion dominante détectée.

        Si aucun visage n'est détecté, retourne:
            {"face_detected": False, "action_units": {},
             "emotions": {}, "dominant_emotion": "neutral"}
    """
    no_face = {
        "face_detected": False,
        "action_units": {},
        "emotions": {},
        "dominant_emotion": "neutral",
    }

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return no_face

    au_results = detect_action_units(frame)

    if not au_results:
        return no_face

    # Use first detected face
    aus = au_results[0]
    emotions = _map_aus_to_emotions(aus)

    # Determine dominant emotion
    dominant = max(emotions, key=emotions.get) if emotions else "neutral"

    return {
        "face_detected": True,
        "action_units": aus,
        "emotions": emotions,
        "dominant_emotion": dominant,
    }
