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
switch nargin
    case [0,1]
        error('Not enough arguments.')
    case 2
        B = 500;
        grid_point = 100;
        concavity = 1;
        increase = 1;
    case 3
        grid_point = 100;
        concavity = 1;
        increase = 1;
    case 4
        concavity = 1;
        increase = 1;
    case 5
        increase = 1;        
end
n = size(X,1);


%% Compute test statistic for original sample

h = BandwidthFixed(X,y,'gaussian','Leave-one-out CV');
% h = LowerBoundBandwidth(X,y,x,h,grid_point,0.25);

[x] = grid_x(X,grid_point,'density');
dlt_index = [];
if size(x,2)> 1
    [x,dlt_index] = drop_grid_without_obs(x,X);
end

[yhat_cc,~,beta_hat_cc] = SCKLS(X,y,'fixed','density','gaussian',grid_point,h,[],concavity,increase,dlt_index);
objective_SCKLS = objective_value_SCKLS(X,y,h,'gaussian',grid_point,x,yhat_cc,beta_hat_cc,1);

[~,~,~,~,yhat_LL_obs,~,yhat_LL,beta_hat_LL] = LL(X,y,'fixed','density','gaussian',grid_point,h,x);
objective_LL = objective_value_SCKLS(X,y,h,'gaussian',grid_point,x,yhat_LL,beta_hat_LL,1);

T = objective_SCKLS - objective_LL;


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

    [yhat_ccB,~,beta_hat_ccB] = SCKLS(X,y_B,'fixed','density','gaussian',grid_point,h,[],concavity,increase,dlt_index);
    objective_SCKLS_B = objective_value_SCKLS(X,y_B,h,'gaussian',grid_point,x,yhat_ccB,beta_hat_ccB,1);
    [~,~,~,~,~,~,yhat_LLB,beta_hat_LLB] = LL(X,y_B,'fixed','density','gaussian',grid_point,h,x);
    objective_LL_B = objective_value_SCKLS(X,y_B,h,'gaussian',grid_point,x,yhat_LLB,beta_hat_LLB,1);

    
    tmp_T_B = objective_SCKLS_B - objective_LL_B;
%     [bb,T,tmp_T_B]
    
    T_B(bb,1) = tmp_T_B;

end


%% Compute p-value

p_value = sum(T<=T_B)/B;


end