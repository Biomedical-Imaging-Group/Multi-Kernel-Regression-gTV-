function [y_test,a] = gTV(x_test,x_tr,y_tr,kernel_param,lambda,hmin)
%gTV kernel estimator (both single and multi case)
%%%%Input
%x_test: testing location
%x_tr: training location. vector of size M (should be in [0,1])
%y_tr: training values. vector of size M
%kernel_param: kernel parameters alpha and gamma. Vector of size 2 by N 
%For the single gTV, set N=1
%lambda: regularization parameter
%hmin: finest grid size
%%%%Output
%y_test: testing values
%a:  kernel coefficients. Vector of size N*M/hmin by 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialization
h=1/8;
[~,N]=size(kernel_param);
a=zeros(N*length(h:h:1),1);
%Multi-res loop
while (h>hmin) 
    %Constructing finer problem
    h=h/2;
    pos=h:h:1;
    Nkernel=N*length(pos);
    anew=upsample(a,2);
    Hmat = Gram_Matrix(x_tr,pos,kernel_param);
    %FISTA Solver
    H=LinOpMatrix(Hmat);
    LS=CostL2([],y_tr);
    F=LS*H;
    F.doPrecomputation=1;
    R_1=CostL1( Nkernel,0);
    G=lambda*R_1;
    FISTA=OptiFBS(F,G);
    FISTA.fista= true;
    FISTA.maxiter=500;
    FISTA.verbose= false;
    FISTA.gam=1/(2*norm(Hmat,2)^2);
    FISTA.run(anew) ;   % run the algorithm
    a=FISTA.xopt;
end
y_test= Kernel_computer(x_test,pos,a,kernel_param);
end
