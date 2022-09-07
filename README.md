[![Build](https://github.com/mahdikooshkbaghi/mavebay/actions/workflows/main.yaml/badge.svg)](https://github.com/mahdikooshkbaghi/mavebay/actions/workflows/main.yaml)

# TODO list for MAVEBAY
Bayesian Version of MAVENN1.0

- [x] The SVI is working both on the abeta and TDP-43 additive inferences.
- [ ] The MCMC on the abeta is working only on small samples.
- [ ] Need to figure out the batch and plate. 
- [ ] Need to modify the `setup.py` to have additional python requirements for the example folder. Check the numpyro github repo for hint.
- [ ] Implementing the skewed-T noise model similar to one we have in MAVENN.
- [ ] K-th order interaction GP map implementation.
    - [x] The K-th=1: which is practically an additive model is working.
- [ ] Measurement process agnostic (MPA) implementation.


## Mathematical explanation for some of TODO items.

### MPA measurement processes

In MPA regression, MAVE-NN directly models the measurement process $p(y|\phi)$. At present, MAVE-NN only supports MPA regression for discrete values of $y$, which are indexed using nonnegative integers. MAVE-NN takes two forms of input for MPA regression. One is a set of (non-unique) sequence-measurement pairs $\{(x_n, y_n)\}_{n=0}^{N-1}$, where $N$ is the total number of independent measurements and $y_n \in  \{0,1,\ldots,Y-1\}$, where $Y$ is the total number of bins. The other is a set of (unique) sequence-count-vector pairs $\{(x_m, c_m)\}_{m=0}^{M-1}$, where $M$ is the total number of unique sequences in the data set, and $\vec{c}_m = (c_{m0}, c_{m1}, \ldots, c_{m(Y-1)})$ lists, for each value of $y$, the number of times $c_{my}$ that the sequence $\vec{x}_m$ was observed in bin $y$. MPA measurement processes are computed internally using the latter representation via

$$p(y|\phi) = \frac{w_y(\phi)}{\sum_{y'} w_{y'}(\phi)}$$

$$w_y(\phi) = \exp \left[ a_y + \sum_{k=0}^{K-1} b_{yk} \tanh(c_{yk} \phi + d_{yk}) \right]~~~~~~
$$

where $K$ is the number of hidden nodes per value of $y$. The trainable parameters of this measurement process are thus $\eta = \{a_y, b_{yk}, c_{yk}, d_{yk}\}$. 