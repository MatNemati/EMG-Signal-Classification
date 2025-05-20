# EMG Signal Classification - Wrist vs Arm

This project implements a basic EMG signal classification pipeline using MATLAB.  
The goal is to classify whether a given EMG signal segment corresponds to **wrist** or **arm** activity, without prior knowledge of the recording source.

---

## Overview

- **Filtering**: Raw EMG signals are filtered using a 2nd-order Butterworth bandpass filter (cut-off: 20–450 Hz).
- **Bias Removal**: The mean of each signal is subtracted.
- **Segmentation**: From each of 10 recordings per class (wrist/arm), the middle 10 seconds (samples 11000–21000) are selected, then divided into 1-second windows.
- **Feature Extraction**: Six time-domain features are extracted from each window.
- **Feature Selection**: Based on scatter plots, `Max` and `Energy` are chosen as the best discriminative features.
- **Classification**: A KNN classifier with k=7 is trained using 5-fold cross-validation.
- **Distance Metrics**: Euclidean, Manhattan, and Minkowski distances are evaluated.
- **Results**: Custom KNN implementation is compared to MATLAB's built-in classifier.

---

## Structure

```
|-- Load Files and Filter
|-- Create 100 Samples per Class
|-- Feature Extraction
|-- Feature Selection (Scatter Plot)
|-- Classifier with KNN
|-- Evaluation (Accuracy & Error)
```

---

## Classification Details

- **Cross-Validation**: 5-Fold (randomized, non-overlapping)
- **Distance Metrics Tested**:
  - Euclidean
  - Manhattan
  - Minkowski
- **Final k Value**: 7 (odd number for binary classification)
- **Evaluation**: Accuracy is computed in each fold and averaged.

---

## Accuracy (Example Results)

| Metric     | Accuracy |
|------------|----------|
| Euclidean  | 84.5%      |
| Manhattan  | 85%      |
| Minkowski  | 83%      |

---

## Tools Used

- **Language**: MATLAB
- **Classifier**: Custom KNN + MATLAB built-in
- **Data**: 20 EMG recordings (10 wrist, 10 arm)

---

## Notes

- Best features: `Max`, `Energy`
- Sample size per class: 100 (total 200 samples)
- Visualization: Enabled through scatter plots in MATLAB

---

## Author

Matin Nemati  
Electrical Engineering Student  
Isfahan University of Technology

---

## License

This project is for academic use only. Feel free to reuse the code with proper credit.