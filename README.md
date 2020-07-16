# Multi-Kernel Regression with Sparsity Constraint

Overview:

    i)  MATLAB implementation of learning using multiple kernels with gTV regularization 

    ii) Comparison with other kernel methods in a simple numerical example.

Any usage of these codes is allowed as long as the user cites the following article:

    S. Aziznejad, M. Unser, "An L1 Representer Theorem for Multiple-Kernel Regression," arXiv:1811.00836 [cs.LG]

Requirements: 

    i) GlobalBioIm library: https://github.com/Biomedical-Imaging-Group/GlobalBioIm

    ii) SimpleMKL package:  http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html

Discription: 

    Example.m : A simple illustrative example of learning a 1D function from its noisy samples.
    GT.m      : The ground-truth function in our example.
    Kernel Estimators: A folder that contains our implementation of four kernel estimators:   
        i)   L2RKHS.m: RKHS with L2 regularization. 
        ii)  L1RKHS.m: RKHS with L1 regularization.
        iii) MKL.m: Multiple kernel learning using SimpleMKL algorithm.
        iv)  gTV.m: gTV-based kernel regression. Both single and multiple kernel scenarios are included.
    Auxilary Functions: A folder that contains the following auxilary functions that are required in our implementations:
        i)   CrossVal.m:    A cross-validation scheme for tuning the hyper-parameters of each method
        ii)  Gram_Matrix.m: Computing the Gramian matrix for both singel and multi-kernel schemes.
        iii) Kernel_computer.m: Computing the learned kernel expansion at a given series of points.
        iv)  Kernel.m: The parametric family of Super-exponential kernels that is used in our example. 
        
Contact information: 
    For more information, please send an e-mail to shayan.aziznejad@epfl.ch or sh.aziznejad@gmail.com
        
