function z = Kernel_computer(x,pos,a,kernel_param)
%%%%%%%Computing the output of the learned kernel estimator%%%%%%
%%%%Input 
%x: Input.
%pos: Kernel positions. vector of size K
%a:  kernel coefficients. Vector of size N*K
%kernel_param: kernel parameters alpha and gamma. Matrix of size 2 by N
%%%%Output
%z: Output.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z=zeros(size(x));
[~,N]=size(kernel_param);
K=length(pos);
a=reshape(a,N,K);
for k=1:K
    for n=1:N
        alpha=kernel_param(1,n);
        gamma=kernel_param(2,n);
        z= z+ a(n,k)* Kernel(x,pos(k) , alpha,gamma);
    end
end
end
