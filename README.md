# Raphael

Module Python d'analyse des émotions basé sur le **FACS** (Facial Action Coding System).

## Fonctionnalités

- **Détection de visage** via MediaPipe Face Mesh (468 landmarks faciaux)
- **Calcul des Action Units (AU)** : AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU12, AU14R, AU15, AU17, AU20, AU23, AU24, AU25, AU26, AU27
- **Reconnaissance d'émotions** à partir des combinaisons d'AUs : joie, tristesse, surprise, peur, colère, dégoût, mépris

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```python
import cv2
from raphael import analyze_emotion

# Charger une image
frame = cv2.imread("photo.jpg")

# Analyser les émotions
result = analyze_emotion(frame)

print(result["face_detected"])      # True / False
print(result["action_units"])       # {"AU1": 0.3, "AU12": 0.8, ...}
print(result["emotions"])           # {"happiness": 0.85, ...}
print(result["dominant_emotion"])   # "happiness"
```

## Détection des Action Units

```python
from raphael import detect_action_units

aus = detect_action_units(frame)
# [{"AU1": 0.12, "AU2": 0.05, "AU6": 0.78, "AU12": 0.85, ...}]
```

## Tests

```bash
pytest tests/
```
