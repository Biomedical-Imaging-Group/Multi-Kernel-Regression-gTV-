function [y] = Kernel(x,x0,alpha,gamma)
%Exponential Kernel 
% alpha = Order 
% Gamma = Dispersion (inversely proportional to width)
y=exp(-gamma*abs(x-x0).^alpha);
end

