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

Check run_example.py in order to see how easy is to use AdaCC1 and/or AdaCC2.
