# Sign Language Detection Using Deep Learning and MediaPipe Holistic

**MSc Computer Science Project — Sambalpur University, 2026**
*Ruchika Mayee Barik | Supervised by Dr. Amiya Kumar Sahu*

---

## What This Project Does

A real-time sign language detection system that recognises hand gestures from a standard webcam and speaks them aloud. No specialised hardware required — just a laptop camera and Python.

The system currently recognises **3 Indian Sign Language gestures**:
- **Good** (thumbs up)
- **Hello** (open-palm wave)
- **Thanks** (hand to chest)

It extracts 1,662 keypoints per frame using Google's MediaPipe Holistic framework, feeds sequences of 30 frames into a stacked LSTM neural network, and outputs the detected sign with a probability confidence bar and text-to-speech announcement.

---

## System Architecture

```
Webcam Frame
    │
    ▼
MediaPipe Holistic (1,662 keypoints/frame)
    │
    ▼
Sequence Buffer (30 frames)
    │
    ▼
LSTM Neural Network → Probability Distribution
    │
    ▼
Detection Filters (hand visibility, frame agreement, confidence gap, cooldown)
    │
    ▼
Display + Text-to-Speech Output
```

**Keypoint breakdown per frame:**
| Body Part | Landmarks | Values |
|-----------|-----------|--------|
| Pose | 33 | 132 (x, y, z + visibility) |
| Face Mesh | 468 | 1,404 (x, y, z) |
| Left Hand | 21 | 63 (x, y, z) |
| Right Hand | 21 | 63 (x, y, z) |
| **Total** | | **1,662** |

---

## Model Architecture

Stacked LSTM with BatchNormalization and L2 regularisation:

| Layer | Units | Notes |
|-------|-------|-------|
| LSTM | 128 | return_sequences=True, L2 |
| BatchNorm + Dropout(0.3) | — | |
| LSTM | 256 | return_sequences=True, L2 |
| BatchNorm + Dropout(0.3) | — | |
| LSTM | 128 | return_sequences=False, L2 |
| BatchNorm + Dropout(0.2) | — | |
| Dense | 128 | ReLU, L2 |
| BatchNorm + Dropout(0.2) | — | |
| Dense | 64 | ReLU |
| Dense (output) | N signs | Softmax |

**Training configuration:**
- Optimizer: Adam (lr=0.001, ReduceLROnPlateau)
- Loss: Categorical Cross-Entropy
- EarlyStopping: patience=30, restores best weights
- Data augmentation: 2× Gaussian noise copies per sequence (triples dataset size)

---

## Results

| Sign | Precision | Recall | F1-Score | Test Samples |
|------|-----------|--------|----------|--------------|
| Good | 1.00 | 1.00 | 1.00 | 68 |
| Hello | 1.00 | 1.00 | 1.00 | 45 |
| Thanks | 1.00 | 1.00 | 1.00 | 45 |
| **Overall** | **1.00** | **1.00** | **1.00** | **158** |

> **Note:** 100% accuracy reflects performance on a 3-class, single-signer, controlled-environment dataset. This is a working prototype, not a production system. See [Limitations](#limitations) below.

---

## Installation

### Prerequisites
- Python 3.10
- Windows 11 (text-to-speech uses Windows Speech API; see Limitations)
- Webcam

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/sign-language-detection.git
cd sign-language-detection
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Usage

### 1. Collect Training Data

```bash
python src/collect_data.py
```

Edit `current_action` inside the script to set which sign to record. The script saves keypoint sequences to `MP_Data/`.

### 2. Train the Model

```bash
python src/train.py
```

Loads all sequences from `MP_Data/`, applies augmentation, trains the LSTM, and saves the best model to `models/best_action.h5`.

### 3. Run Real-Time Detection

```bash
python src/predict.py
```

Opens the webcam feed. Perform a sign gesture and hold it for ~1 second. Press `q` to quit, `c` to clear the sentence.

---

## Adding New Signs

The system is fully extensible. To add a new sign:

1. Run `collect_data.py` with `current_action = 'your_new_sign'`
2. Record at least 50–100 sequences
3. Run `train.py` — it auto-discovers all signs in `MP_Data/`
4. Run `predict.py` — the new sign is automatically included

No changes to the model architecture or core code required.

---

## Pretrained Model

The pretrained model (`models/best_action.h5`) recognises: **good, hello, thanks**

It was trained on self-recorded sequences under consistent indoor lighting. Performance on different signers, lighting conditions, or camera angles will vary.

---

## Limitations

These are known and documented, not overlooked:

| Limitation | Detail |
|---|---|
| Vocabulary | Only 3 signs. Expanding requires re-recording and retraining. |
| Single signer | Trained on one person's recordings. Cross-signer generalisation is untested. |
| Windows-only TTS | Text-to-speech uses Windows PowerShell Speech API. Linux/Mac users need to replace the `speak()` function. |
| No GPU required | Runs on CPU at ~15fps. This is a feature for accessibility but limits real-time speed. |
| Controlled environment | Recorded under consistent lighting and background. Outdoor or cluttered backgrounds may reduce accuracy. |
| 100% accuracy caveat | Three-class, single-signer evaluation. This number does not represent generalised accuracy. |

---

## Project Structure

```
sign-language-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── Sign_Language_Detection.ipynb   # Full pipeline walkthrough notebook
├── src/
│   ├── collect_data.py             # Webcam data collection
│   ├── train.py                    # Model training pipeline
│   ├── predict.py                  # Real-time inference
│   └── utils.py                    # Shared helper functions
├── models/
│   ├── best_action.h5              # Pretrained model weights
│   └── README.md
├── data/
│   └── sample/                     # Sample .npy sequences for testing
├── results/
│   ├── training_history.png
│   └── classification_report.txt
└── docs/
    └── architecture.png
```

> **Note:** The full `MP_Data/` training dataset is not included in this repository due to size (thousands of `.npy` files). To train from scratch, use `collect_data.py` to record your own sequences.

---

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.10.8 |
| TensorFlow / Keras | 2.12.0 |
| MediaPipe | 0.10.5 |
| OpenCV | 4.13.0 |
| NumPy | 1.23.5 |
| scikit-learn | 1.2.0 |
| Matplotlib | 3.7.x |
| SciPy | 1.10.x |

---

## References

1. Mitra, S. & Acharya, T. (2007). Gesture Recognition: A Survey. *IEEE Transactions on Systems, Man, and Cybernetics*, 37(3), 311–324.
2. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
3. Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv:1906.08172*.
4. Koller, O. et al. (2017). Re-Sign: Re-Aligned End-to-End Sequence Modelling with Deep Recurrent CNN-HMMs. *CVPR 2017*.
5. Renotte, N. (2021). Action Detection for Sign Language. GitHub / YouTube tutorial series.
6. Al-Qurishi, M. et al. (2021). Deep Learning for Sign Language Recognition. *IEEE Access*, 9, 126917–126951.

---

## License

MIT License — see `LICENSE` for details.

---

*Department of Computer Science and Applications, Sambalpur University, Odisha*
