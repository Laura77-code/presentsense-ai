# Model and Dataset Notes

## Dataset

The expression recognition model was trained using FER2013.

Dataset link:

- [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset is not included in this repository because of size and licensing considerations.

## Expected Classes

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

## Best Model

The best model was **ResNet18 fine-tuned on FER2013**.

| Experiment | Model | Epochs | Batch Size | Test Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| exp04 | MobileNetV3 fine-tune | 10 | 32 | 49.53% | 42.74% |
| exp04 | MobileNetV3 fine-tune | 20 | 32 | 51.49% | 45.62% |
| exp03 | ResNet18 fine-tune | 10 | 16 | **64.11%** | **60.87%** |

## Why Use `uncertain`?

The model was trained on FER2013, which differs from real webcam footage. To avoid overclaiming predictions, PresentSense labels low-confidence frames as `uncertain` instead of forcing a visible expression label.
