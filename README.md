# Cumulative Cost Sensitive Boosting (AdaCC)

This repo contains AdaCC method. AdaCC method is a dynamic cost-sensitive method which estimates the misclassification costs based on the behavior of the partial ensemble to minimize the balanced error.

The following example showcases how the weighting strategy of AdaCC differs from AdaBoost.

<figure>
  <img src="boost_toy.png" alt="AdaBoost">
</figure>

<figure>
  <img src="adacc1_toy.png" alt="AdaCC1">
</figure>

<figure>
  <img src="adacc2_toy.png" alt="AdaCC2">
</figure>

Check run_example.py in order to see how to use AdaCC1 and/or AdaCC2.

In case you employ this method in your work, use this as a citation point:

@article{iosifidis2022adacc,
  title={AdaCC: cumulative cost-sensitive boosting for imbalanced classification},
  author={Iosifidis, Vasileios and Papadopoulos, Symeon and Rosenhahn, Bodo and Ntoutsi, Eirini},
  journal={Knowledge and Information Systems},
  pages={1--38},
  year={2022},
  publisher={Springer}
}
