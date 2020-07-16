%% Comparing kernel estimators for recovery of a GT function
clc
clear all
close all
font_size = 20; line_width = 3; marker_size = 5;

%MSE Errors in the full-data case
e2=0;e1=0;eMKL=0;egTV=0;emulti=0;
%Sparsity in the full-data case
s2=0;s1=0;smkl=0;ssing=0;smult=0;

%MSE Errors in the missing-data case
e2_missing=0;e1_missing=0;eMKL_missing=0;egTV_missing=0;emulti_missing=0;
%Sparsity in the missing-data case
s2_missing=0;s1_missing=0;smkl_missing=0;ssing_missing=0;smult_missing=0;

%Ground-truth model
t=linspace(0,1,10000)';
gt=@(t) GT(t);
y_test=gt(t);

%Learning Framework Parameters
sigma_noise=0.1;%Gaussian noise std
M=100;          %Training size
Num_fold=5;     %Number of Cross-folding
K=10;           %Number of trials

% Kernel Parameters
gammavec=logspace(1,5,10);       %Different gamma= 1/(2*sigma^2)
lambdavec=logspace(-9,0,10);     %Different lambda
alphaRKHS=2; alphagTV=1.99;      %Gaussian kernels
hmin=2^(-10);                    %Finest grid size in gTV

%% Trial Loop
for k=1:K
    k
%Training dataset missing
    x_data_missing=rand(M,1)*0.8;
    x_data_missing=x_data_missing+0.2*(x_data_missing>0.6);
    y_data_missing=gt(x_data_missing)+ sigma_noise*randn(size(x_data_missing));
    
%Training dataset full
    x_data_full=rand(M,1);
    y_data_full=gt(x_data_full)+ sigma_noise*randn(size(x_data_full));
    
    
%Kernel Estimators

    %L2 RKHS
    method= @(x,x_tr,y_tr,gamma,lambda) L2RKHS(x,x_tr,y_tr,[alphaRKHS;gamma],lambda);    
    %Full
    [ft2,err2_test,err2_tr,a2_opt]=CrossVal(x_data_full,y_data_full,t,y_test,t,Num_fold,method,gammavec,lambdavec);
    %Missing
    [ft2_missing,err2_test_missing,err2_tr_missing,a2_opt_missing]=CrossVal(x_data_missing,y_data_missing,t,y_test,t,Num_fold,method,gammavec,lambdavec);

    %L1 RKHS
    method= @(x,x_tr,y_tr,gamma,lambda) L1RKHS(x,x_tr,y_tr,[alphaRKHS;gamma],lambda);
    %Full
    [ft1,err1_test,err1_tr,a1_opt]=CrossVal(x_data_full,y_data_full,t,y_test,t,Num_fold,method,gammavec,lambdavec);
    %Missing
    [ft1_missing,err1_test_missing,err1_tr_missing,a1_opt_missing]=CrossVal(x_data_missing,y_data_missing,t,y_test,t,Num_fold,method,gammavec,lambdavec);
    
    %Single-gTV
    method= @(x,x_tr,y_tr,gamma,lambda) gTV(x,x_tr,y_tr,[alphagTV;gamma],lambda,hmin);
    %Full
    [ftgTV,errgTV_test,errgTV_tr,agTV_opt]=CrossVal(x_data_full,y_data_full,t,y_test,t,Num_fold,method,gammavec,lambdavec);
    %Missing
    [ftgTV_missing,errgTV_test_missing,errgTV_tr_missing,agTV_opt_missing]=CrossVal(x_data_missing,y_data_missing,t,y_test,t,Num_fold,method,gammavec,lambdavec);
    
    %Multi-gTV
    method= @(x,x_tr,y_tr,h_artificial,lambda) gTV(x,x_tr,y_tr,h_artificial*[alphagTV*ones(size(gammavec));gammavec],lambda,hmin);
    %Full
    [ftmulti,errmulti_test,errmulti_tr,amulti_opt]=CrossVal(x_data_full,y_data_full,t,y_test,t,Num_fold,method,1,lambdavec);
    %Missing
    [ftmulti_missing,errmulti_test_missing,errmulti_tr_missing,amulti_opt_missing]=CrossVal(x_data_missing,y_data_missing,t,y_test,t,Num_fold,method,1,lambdavec);
    
    %SimpleMKL
    method= @(x,x_tr,y_tr,h_artificial,lambda) MKL(x,x_tr,y_tr,h_artificial*gammavec,lambda);
    %Full
    [ftMKL,errMKL_test,errMKL_tr,aMKL_opt]=CrossVal(x_data_full,y_data_full,t,y_test,t,Num_fold,method,1,lambdavec);
    %Missing
    [ftMKL_missing,errMKL_test_missing,errMKL_tr_missing,aMKL_opt_missing]=CrossVal(x_data_missing,y_data_missing,t,y_test,t,Num_fold,method,1,lambdavec);
    
% Coeffs
    %Full
    a2_sorted=(sort(abs(a2_opt)/max(abs(a2_opt)),'descend'));
    a1_sorted=(sort(abs(a1_opt)/max(abs(a1_opt)),'descend'));
    amkl_sorted=(sort(abs(aMKL_opt)/max(abs(aMKL_opt)),'descend'));
    asing_sorted=(sort(abs(agTV_opt)/max(abs(agTV_opt)),'descend'));
    amult_sorted=(sort(abs(amulti_opt)/max(abs(amulti_opt)),'descend'));
    %Missing
    a2_sorted_missing=(sort(abs(a2_opt_missing)/max(abs(a2_opt_missing)),'descend'));
    a1_sorted_missing=(sort(abs(a1_opt_missing)/max(abs(a1_opt_missing)),'descend'));
    amkl_sorted_missing=(sort(abs(aMKL_opt_missing)/max(abs(aMKL_opt_missing)),'descend'));
    asing_sorted_missing=(sort(abs(agTV_opt_missing)/max(abs(agTV_opt_missing)),'descend'));
    amult_sorted_missing=(sort(abs(amulti_opt_missing(:))/max(abs(amulti_opt_missing)),'descend'));
    
%MSE accomulation (need to divide by K at the end)
    %Full
    e2=e2+err2_test;
    e1=e1+err1_test;
    eMKL=eMKL+errMKL_test;
    egTV=egTV+errgTV_test;
    emulti=emulti+errmulti_test;
    %Missing
    e2_missing=e2_missing+err2_test_missing;
    e1_missing=e1_missing+err1_test_missing;
    eMKL_missing=eMKL_missing+errMKL_test_missing;
    egTV_missing=egTV_missing+errgTV_test_missing;
    emulti_missing=emulti_missing+errmulti_test_missing;
%Sparsity accomulation  (need to divide by K at the end)
    %Full
    s2=s2+nnz(a2_sorted>0.1);
    s1=s1+nnz(a1_sorted>0.1);
    smkl=smkl+nnz(amkl_sorted>0.1);
    ssing=ssing+nnz(asing_sorted>0.1);
    smult=smult+nnz(amult_sorted>0.1);
    %Missing
    s2_missing=s2_missing+nnz(a2_sorted_missing>0.1);
    s1_missing=s1_missing+nnz(a1_sorted_missing>0.1);
    smkl_missing=smkl_missing+nnz(amkl_sorted_missing>0.1);
    ssing_missing=ssing_missing+nnz(asing_sorted_missing>0.1);
    smult_missing=smult_missing+nnz(amult_sorted_missing>0.1);
end
%% Displaying MSE and Sparsity
clc;
%Full-data case
    disp(['Results for the full-data case']);
    disp(['L2RKHS: ',' MSE=',num2str(e2/K,2),' Sparsity=',num2str(s2/K,2)]);
    disp(['L1RKHS: ',' MSE=',num2str(e1/K,2),' Sparsity=',num2str(s1/K,2)]);
    disp(['SimpleMKL: ',' MSE=',num2str(eMKL/K,2),' Sparsity=',num2str(smkl/K,2)]);
    disp(['Single-gTV: ',' MSE=',num2str(egTV/K,2),' Sparsity=',num2str(ssing/K,2)]);
    disp(['Multi-gTV: ',' MSE=',num2str(emulti/K,2),' Sparsity=',num2str(smult/K,2)]);
%Missing-data case
    disp(['Results for the missing-data case']);
    disp(['L2RKHS: ',' MSE=',num2str(e2_missing/K,2),' Sparsity=',num2str(s2_missing/K,2)]);
    disp(['L1RKHS: ',' MSE=',num2str(e1_missing/K,2),' Sparsity=',num2str(s1_missing/K,2)]);
    disp(['SimpleMKL: ',' MSE=',num2str(eMKL_missing/K,2),' Sparsity=',num2str(smkl_missing/K,2)]);
    disp(['Single-gTV: ',' MSE=',num2str(egTV_missing/K,2),' Sparsity=',num2str(ssing_missing/K,2)]);
    disp(['Multi-gTV: ',' MSE=',num2str(emulti_missing/K,2),' Sparsity=',num2str(smult_missing/K,2)]);    
%% MSE Plots (last trial)
figure;
%Full
subplot(3,2,1) 
plot(x_data_full, y_data_full, '*','LineWidth', line_width, 'Markersize', 5,'color','red');
hold on;plot(t,y_test, 'LineWidth',line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size);leg = {'Training','GT'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,2);
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ft2,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','RKHS $L_2$'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,3);
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ft1,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','RKHS $L_1$'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,4); 
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ftMKL,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','SimpleMKL'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,5);
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ftgTV,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','Single gTV'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,6);
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ftmulti,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','Multi gTV'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

%Missing
figure
subplot(3,2,1) 
plot(x_data_missing, y_data_missing, '*','LineWidth', line_width, 'Markersize', 5,'color','red');
hold on;plot(t,y_test, 'LineWidth',line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size);leg = {'Training','GT'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,2); 
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ft2_missing,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','RKHS $L_2$'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,3); 
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ft1_missing,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','RKHS $L_1$'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,4); 
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ftMKL_missing,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','SimpleMKL'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,5); 
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ftgTV_missing,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','Single gTV'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

subplot(3,2,6); 
plot(t,y_test,  'LineWidth', line_width,'color','black');
ax = gca; set(ax, 'FontSize', font_size); hold on;
plot(t, ftmulti_missing,'-.', 'Linewidth', line_width,'color','blue');
leg = {'GT','Multi gTV'};
legend(ax, leg,'Interpreter','latex', 'Location', 'Southeast');

%% Sparsity Plots in the full-data case (last trial)
num=100;
amkl_sorted=[amkl_sorted ;zeros((num-length(amkl_sorted)),1)];
figure
subplot(5,1,1),stem(a2_sorted(1:num),'LineWidth', line_width),title('L2-RKHS')
ax = gca; set(ax, 'FontSize', font_size);

subplot(5,1,2),stem(a1_sorted(1:num),'LineWidth', line_width),title('L1-RKHS (LASSO)')
ax = gca; set(ax, 'FontSize', font_size);

subplot(5,1,3),stem(amkl_sorted(1:num),'LineWidth', line_width),title('SimpleMKL')
ax = gca; set(ax, 'FontSize', font_size);

subplot(5,1,4),stem(asing_sorted(1:num),'LineWidth', line_width),title('Single gTV')
ax = gca; set(ax, 'FontSize', font_size);

subplot(5,1,5),stem(amult_sorted(1:num),'LineWidth', line_width),title('Multi gTV')
ax = gca; set(ax, 'FontSize', font_size);
