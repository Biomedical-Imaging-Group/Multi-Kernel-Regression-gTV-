function [y_test,a] = L1RKHS(x_test,x_tr,y_tr,kernel_param,lambda)
%RKHS L1 kernel estimator (generalized LASSO)
%%%%Input
%x_test: testing location
%x_tr: training location. vector of size M
%y_tr: training values. vector of size M
%kernel_param: kernel parameters alpha and gamma. Vector of size 2 by 1
%lambda: regularization parameter
%%%%Output
%y_test: testing values
%a:  kernel coefficients. Vector of size M by 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hmat= Gram_Matrix(x_tr,x_tr,kernel_param);
% FISTA for solving the LASSO problem
H=LinOpMatrix(Hmat);
LS=CostL2([],y_tr);
F=LS*H;
F.doPrecomputation=1;
R_1=CostL1(length(x_tr),0);
G=lambda*R_1;
FISTA=OptiFBS(F,G);
FISTA.fista= true;
FISTA.maxiter=500;
FISTA.verbose= false;
FISTA.gam=1/(2*norm(Hmat,2)^2);
FISTA.run(zeros(length(x_tr),1)) ;   % run the algorithm
a=FISTA.xopt;
%End of FISTA
y_test= Kernel_computer(x_test,x_tr,a,kernel_param);
end