# Known Limitations

This document lists known limitations of the current system. These are acknowledged design constraints, not oversights.

## 1. Vocabulary Size

The current model recognises **3 signs only** (Good, Hello, Thanks). This is a proof-of-concept implementation. Expanding the vocabulary requires:
- Recording additional sequences using `collect_data.py`
- Retraining the model using `train.py`
- No changes to the core architecture are needed

## 2. Single Signer

All training data was recorded by one person under consistent indoor lighting and background. The model has **not been tested across different signers**. Cross-signer generalisation is expected to be poor without:
- Recording data from multiple signers
- Applying transfer learning from a larger sign language dataset
- Data normalisation relative to body proportions

## 3. Windows-Only Text-to-Speech

The `speak()` function in `utils.py` uses the **Windows PowerShell Speech API**. It will not work on Linux or macOS. To use on other platforms, replace the `speak()` function with an alternative:
- `pyttsx3` (cross-platform)
- `espeak` (Linux)
- `gtts` + `playsound` (cross-platform, requires internet)

## 4. Controlled Environment

The system was developed and tested under consistent indoor lighting with a plain background. Performance will degrade under:
- Poor or variable lighting
- Cluttered or moving backgrounds
- Partial occlusion of hands or face

## 5. Accuracy Caveat

The reported **100% test accuracy** reflects performance on a 3-class, single-signer, held-out test set derived from the same recording session as training data. This number should not be interpreted as a measure of real-world generalisation. It indicates that the model has learned the specific gesture patterns of the training signer in the training environment.

## 6. Processing Speed

The system runs at approximately **15 fps on a standard CPU** without GPU acceleration. This is sufficient for demonstration but may lag on older hardware.

## 7. No Continuous Signing Support

The system is designed for **isolated sign recognition** — one sign at a time, with deliberate pauses between signs. It does not support continuous, sentence-level signing fluency.
