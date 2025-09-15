# PINNs-and-Symbolic-regression-for-solving-kinetic-ODEs
PINNs and Symbolic regression for solving kinetic ODEs.

In this example we use Physics-Informed Neural Networks (PINNs) and symbolic regression to uncover parameters of Ordinary Differential Equation(s) that describe the kinetic process of ageing of cellulose. 

In the first example we use Ekenstam ODE:

$\frac{𝑑𝐷𝑃}{𝑑𝑡}=−𝑘⋅𝐷𝑃^2$

where:

$𝑘=𝐴⋅𝑒^{(−\frac{𝐸}{𝑅𝑇})}$

$𝐷𝑃_0=𝐷𝑃(0)$

In the second example we solve Emseley system of ODEs:

$\frac{𝑑𝐷𝑃}{𝑑𝑡}=−𝑘_1⋅𝐷𝑃^2$

$\frac{(𝑑𝑘_1)}{𝑑𝑡}=−𝑘_2⋅𝑘_1$

$𝐷𝑃_0=𝐷𝑃(0)$
$𝑘_{1_{0}}=𝑘_1 (0)$


## Implementation

PINNs are implemented using Python and Tensorflow, using Adam opimizer. 

All the hyperparameters can be found https://arxiv.org/abs/2504.03484 
