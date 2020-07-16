function [y_test,a] = L2RKHS(x_test,x_tr,y_tr,kernel_param,lambda)
%RKHS L2 kernel estimator 
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
M=length(x_tr);
Hmat= Gram_Matrix(x_tr,x_tr,kernel_param);
a=(Hmat+lambda*eye(M))\(y_tr); %Finding Kernel Coefficients
y_test= Kernel_computer(x_test,x_tr,a,kernel_param);
end
