function [y_test,a] = MKL(x_test,x_tr,y_tr,gammavec,lambda)
%SimpleMKL with Gaussian kernels
%%%%Input
%x_test: testing location
%x_tr: training location. vector of size M
%y_tr: training values. vector of size M
%gammavec: Each element is gamma_n=1/(2*width_n^2). Vector of size N. . 
%lambda: regularization parameter
%C: SVM Hyperparameter
%%%%Output
%y_test: testing values
%a:  kernel coefficients. Vector of size M by 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SimpleMKL Parameters
C=1e4;
verbose=0;
options.algo='svmreg';
options.seuildiffsigma=1e-3;
options.seuildiffconstraint=0.01;
options.seuildualitygap=0.01;
options.goldensearch_deltmax=1e-2;
options.numericalprecision=1e-8;
options.stopvariation=0;
options.stopKKT=0;
options.stopdualitygap=1;
options.firstbasevariable='first';
options.nbitermax=100;
options.seuil=0;
options.seuilitermax=10;
options.miniter=0;
options.verbosesvm=0;
options.svmreg_epsilon=0.01;
options.efficientkernel=0;
optionK.pow=0;
kernelt={'gaussian'};
variablevec={'all'};

%Feeding vector of widths
kerneloptionvect={sqrt(1./(2*gammavec))}; 

%Feeding Hyper-parameter
options.lambdareg = lambda;

%SimpleMKL Solver
dim=size(x_tr,2);
[kernel,kerneloptionvec,optionK.variablecell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
[K]=mklbuildkernel(x_tr,kernel,kerneloptionvec,[],[],optionK);
[K,optionK.weightK]=WeightK(K);
[beta,w,b,posw,~,~] = mklsvm(K,y_tr,C,options,verbose);
%Computing the outputs
kerneloption.matrix=mklbuildkernel(x_test,kernel,kerneloptionvec,x_tr(posw,:),beta,optionK);
y_test=svmval([],[],w,b,'numerical',kerneloption);
a=w;
end