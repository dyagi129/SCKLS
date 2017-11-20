%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function to test shape constraints imposed which is porposed
% by Yagi et al (2016).
%
%
% Input arguments:
%       X:              observed input
%       y:              observed output
%       B:              number of Bootstrap iterations
%       grid_point:     number of grid points for the estimation
%       concavity:      0:convex, 1:concave
%       increase:       0:decreasing, 1:increasing
%
%
% Output arguments:
%       p_value:        p-value of hypothesis test (H_0: constraints are true, H_1: constraints are wrong)
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [p_value] = shape_test(X,y,B,grid_point,concavity,increase)


%% Initialization
switch nargin,
    case [0,1],
        error('Not enough arguments.')
    case 2,
        B = 500;
        grid_point = 100;
        concavity = 1;
        increase = 1;
    case 3,
        grid_point = 100;
        concavity = 1;
        increase = 1;
    case 4,
        concavity = 1;
        increase = 1;
    case 5,
        increase = 1;        
end
n = size(X,1);


%% Compute test statistic for original sample

h = BandwidthFixed(X,y,'gaussian','Leave-one-out CV');
% h = LowerBoundBandwidth(X,y,x,h,grid_point,0.25);

[x] = grid_x(X,grid_point,'density');
[x,dlt_index] = drop_grid_without_obs(x,X);

% figure()
% hold on
% plot(X(:,1),X(:,2),'.')
% plot(x(:,1),x(:,2),'.')
% xlabel('x_1')
% ylabel('x_2')
% xlim([-1,7])
% ylim([-1,7])
% title('Evaluation points')

% figure()
% hold on
% plot(X(:,1),X(:,2),'.')
% plot(x(:,1),x(:,2),'.')

% [yhat_cc,tmp,beta_hat_cc] = SCKLS(X,y,'fixed','equal','gaussian',grid_point,h,X,concavity,increase);
% objective_SCKLS = objective_value_SCKLS(X,y,h,'gaussian',grid_point,X,yhat_cc,beta_hat_cc,1);
% 
% [tmp,tmp,tmp,tmp,yhat_LL,beta_hat_LL] = LL(X,y,'fixed','equal','gaussian',grid_point,h,X);
% yhat_LL_obs = yhat_LL;
% objective_LL = objective_value_SCKLS(X,y,h,'gaussian',grid_point,X,yhat_LL,beta_hat_LL,1);

[yhat_cc,tmp,beta_hat_cc] = SCKLS(X,y,'fixed','density','gaussian',grid_point,h,[],concavity,increase,dlt_index);
objective_SCKLS = objective_value_SCKLS(X,y,h,'gaussian',grid_point,x,yhat_cc,beta_hat_cc,1);

[tmp,tmp,tmp,tmp,yhat_LL_obs,tmp,yhat_LL,beta_hat_LL] = LL(X,y,'fixed','density','gaussian',grid_point,h,x);
objective_LL = objective_value_SCKLS(X,y,h,'gaussian',grid_point,x,yhat_LL,beta_hat_LL,1);



T = objective_SCKLS - objective_LL;

figure()
hold on
plot3(x(:,1),x(:,2),yhat_cc,'.')
plot3(x(:,1),x(:,2),yhat_LL,'.')
plot3(X(:,1),X(:,2),y,'.')
legend('SCKLS','LL')
xlabel('X_1 (Labor)')
ylabel('X_2 (Capital)')
zlabel('y')

%% Bootstrap to recover the distribution of test statistics

% 1. Compute residuals of LL
epsilon_hat = y-yhat_LL_obs;

% 2. Simulate Bootstrapping Sample
T_B = zeros(B,1);
parfor bb = 1:B
% for bb = 1:B
    u_B = randi([0,1],n,1);
    u_B(u_B==0) = -1;   % Randomly generate 1 or -1
    y_B = epsilon_hat.*u_B;   

%     [yhat_ccB,alpha_hat_ccB,beta_hat_ccB] = SCKLS(X,y_B,'fixed','equal','gaussian',grid_point,h,X,concavity,increase);
%     objective_SCKLS_B = objective_value_SCKLS(X,y_B,h,'gaussian',grid_point,X,yhat_ccB,beta_hat_ccB,1);
%     [tmp,tmp,tmp,tmp,yhat_LL_B,beta_hat_LL_B] = LL(X,y_B,'fixed','equal','gaussian',grid_point,h,X);
%     objective_LL_B = objective_value_SCKLS(X,y_B,h,'gaussian',grid_point,X,yhat_LL_B,beta_hat_LL_B,1);

    [yhat_ccB,tmp,beta_hat_ccB] = SCKLS(X,y_B,'fixed','density','gaussian',grid_point,h,[],concavity,increase,dlt_index);
    objective_SCKLS_B = objective_value_SCKLS(X,y_B,h,'gaussian',grid_point,x,yhat_ccB,beta_hat_ccB,1);
    [tmp,tmp,tmp,tmp,tmp,tmp,yhat_LLB,beta_hat_LLB] = LL(X,y_B,'fixed','density','gaussian',grid_point,h,x);
    objective_LL_B = objective_value_SCKLS(X,y_B,h,'gaussian',grid_point,x,yhat_LLB,beta_hat_LLB,1);

    
    tmp_T_B = objective_SCKLS_B - objective_LL_B;
    [bb,T,tmp_T_B]
    
    T_B(bb,1) = tmp_T_B;
    
%     figure()
%     hold on
%     plot3(x(:,1),x(:,2),yhat_ccB,'.')
%     plot3(x(:,1),x(:,2),yhat_LLB,'.')
%     plot3(X(:,1),X(:,2),y_B,'.')
%     legend('SCKLS','LL')
%     xlabel('X_1 (Labor)')
%     ylabel('X_2 (Capital)')
%     zlabel('y')
end


%% Compute p-value

p_value = sum(T<=T_B)/B;


end