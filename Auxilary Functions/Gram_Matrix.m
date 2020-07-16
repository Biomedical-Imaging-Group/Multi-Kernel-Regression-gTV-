function G = Gram_Matrix (x_tr,pos,kernel_param)
%%%%%%Computing the Grammian matrix G=[k(x_m,z_n)]_{m,n}%%%%%%
%%%%Input
%x_tr: Training data location. Vector of size M
%pos:  Kernel positions. vector of size K
%kernel_param: super-exponential kernel parameters alpha and gamma. Matrix of size 2 by N
%%%%Output
%G: Gram matrix of size M by NK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=length(x_tr);
K=length(pos);
[~,N]=size(kernel_param);
G=zeros(M,K*N);
for m=1:M
    for k=1:K
        for n=1:N
            alpha=kernel_param(1,n);
            gamma=kernel_param(2,n);
            G(m,(k-1)*N+n)=  Kernel(x_tr(m),pos(k),alpha,gamma);
        end
    end
end
end
