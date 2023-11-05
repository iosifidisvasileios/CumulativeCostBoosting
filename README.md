# AdaCC: Adaptive Cost-sensitive Boosting for Imbalanced Data

## Overview
AdaCC is a novel cost-sensitive boosting approach designed to address the challenge of class imbalance in machine learning. Traditional supervised learning models often exhibit bias towards the majority class, leading to under-performance in minority classes. Cost-sensitive learning attempts to mitigate this issue by treating classes differently, employing a fixed misclassification cost matrix provided by users. However, manually tuning these parameters can be daunting and might lead to suboptimal results if not done accurately.

In this work, we introduce AdaCC, a groundbreaking method that dynamically adjusts misclassification costs over boosting rounds based on the model's performance. Unlike conventional approaches, AdaCC eliminates the need for fixed misclassification cost matrices, offering a parameter-free solution. By leveraging the cumulative behavior of the boosting model, AdaCC automatically adapts misclassification costs for subsequent boosting rounds, ensuring optimal balance between classes and enhancing predictive accuracy.


The following example showcases how the weighting strategy of AdaCC differs from AdaBoost. Feel free to also read the technical paper of this approach ["AdaCC: cumulative cost-sensitive boosting for imbalanced classification"](https://link.springer.com/article/10.1007/s10115-022-01780-8)

<figure>
  <img src="boost_toy.png" alt="AdaBoost">
</figure>

<figure>
  <img src="adacc1_toy.png" alt="AdaCC1">
</figure>

<figure>
  <img src="adacc2_toy.png" alt="AdaCC2">
</figure>

## Key Features
- **Dynamic Cost Adjustment:** AdaCC dynamically modifies misclassification costs in response to the boosting model's performance, optimizing class balance without the need for manual parameter tuning.

- **Parameter-Free Solution:** Eliminates the complexity of defining fixed misclassification cost matrices, providing a hassle-free experience for users without requiring domain knowledge.

- **Theoretical Guarantees:** AdaCC comes with theoretical guarantees regarding training error, ensuring the reliability and robustness of the boosting model across various datasets.

## How to Use
1. **Installation**
- pip install cumulative-cost-boosting

2. **Usage**

```
from cumulative_cost_boosting import AdaCC

clf = AdaCC(n_estimators=100, algorithm='AdaCC1')
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
```

## Example

A detailed example is displayed in the `run_example.py` file.

## Contributions and Issues
Contributions and feedback are welcome. If you encounter any issues or have suggestions for improvement, please feel free to create an issue in the repository or submit a pull request.

**Note:** AdaCC is a cutting-edge solution for handling class imbalance, ensuring accurate and fair predictions in machine learning tasks. Thank you for considering AdaCC for your imbalanced data challenges. Let's empower your models together!




