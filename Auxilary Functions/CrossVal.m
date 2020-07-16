function [ft,err_test,err_tr,a_opt]=CrossVal(x_data,y_data,x_test,y_test,t,Num_fold,method,H1vec,H2vec)
%%%%%%%Cross-validation for tuning hyper-parameters%%%%%%%
%%%%Input
%(x_data,y_data): training data
%(x_test,y_test): testing data
%t: input variable
%Num_fold: Number of folding in Cross-validation
%method: gTV,L1RKHS,L2RKHS,MKL. Should be given as handle
%H1vec,H2vec: Vectors of hyper-parameters.
%%%%Output
%ft: output of the kernel estimator to the vector t
%err_test: testing error
%err_tr: training error
%a_opt: kernel coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind=crossvalind('Kfold',x_data,Num_fold);
err_va=inf; 
for h1=H1vec
    for h2=H2vec
        err_crossval=0;
        for nfold=1:Num_fold
            ind_va=(ind == nfold); x_va=x_data(ind_va); y_va=y_data(ind_va);
            ind_tr=~ind_va; x_tr=x_data(ind_tr); y_tr=y_data(ind_tr);
            [z_va,~] = method(x_va,x_tr,y_tr,h1,h2); 
            err_crossval=err_crossval+norm(z_va-y_va,2);
        end
        err_crossval=err_crossval/Num_fold;
        if err_crossval<err_va
            err_va=err_crossval;
            h1_opt=h1;
            h2_opt=h2;
        end
    end
end
%Solving for the optimum hyper-parameters
X=[ t ;x_data ;x_test];
[Z,a_opt] = method(X,x_data,y_data,h1_opt,h2_opt);

%Reporting outputs
ft=Z(1:length(t));

z_tr=Z(length(t)+1:length(t)+length(x_data));
err_tr=20*log10(norm(z_tr-y_data,2)/sqrt(length(x_data)));

z_test=Z(length(t)+length(x_data)+1:end);
err_test=20*log10(norm(z_test-y_test,2)/sqrt(length(x_test)));
end


