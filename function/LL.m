%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function of Local Linear Kernel with Leave-one-out cross
% validation for the bandwidth selection.
%
%
% Input arguments:
%       X:              observed input (can be multiple columns)
%       y:              observed output (have to be single column)
%       type_bandwidth: type of bandwidth ('fixed','variable')
%                       where I use KNN for variable bandwidth
%       kernel:         type of kernel ('gaussian','uniform','epanechnikov')
%
%
% Output arguments:
%       yhat:           functional estimate on each grid points
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [yhat,alpha_hat,beta_hat,x,yhat_obs,beta_hat_obs,yhat_eval,beta_hat_eval] = LL(X,y,type_bandwidth,type_grid,kernel,grid_point,bandwidth,X_eval)


%% Error check
n = size(X,1);
d = size(X,2);


switch nargin,
    case [0,1],
        error('Not enough arguments.')
    case 2,
        type_bandwidth = 'fixed';
        type_grid = 'equal';
        kernel = 'gaussian';
        grid_point = 100;
        bandwidth = std(X)*n^(-1/(4+d));
        X_eval = [];
    case 3,
        kernel = 'gaussian';
        type_grid = 'equal';
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        X_eval = [];
    case 4,
        type_grid = 'equal';
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        X_eval = [];
    case 5,
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        X_eval = [];
    case 6,
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        X_eval = [];
    case 7,
        X_eval = [];
end

if strcmp(type_bandwidth,'fixed') + strcmp(type_bandwidth,'variable') == 0,
    error('"type_bandwidth" has wrong name. Choose "fixed" or "variable".')
end
if strcmp(kernel,'gaussian') + strcmp(kernel,'uniform') + strcmp(kernel,'epanechnikov') == 0,
    error('"kernel" has wrong name. Choose "gaussian", "uniform" or "epanechnikov".') 
end
if strcmp(type_grid,'equal') + strcmp(type_grid,'percentile') + strcmp(type_grid,'density') == 0,
    error('"type_grid" has wrong name. Choose "equal", "percentile" or "density".')
end


%% Standarize data
X_tilde=(X-repmat(mean(X),n,1))./repmat(std(X),n,1);


%% Get grid
if grid_point == 0
    x = [];
    x_tilde = [];
else
    [x] = grid_x(X,grid_point,type_grid);
    x_tilde=(x-repmat(mean(X),length(x),1))./repmat(std(X),length(x),1);
end
if isempty(X_eval)
    X_eval_tilde = [];
else
    X_eval_tilde=(X_eval-repmat(mean(X),length(X_eval),1))./repmat(std(X),length(X_eval),1);
end
%% Compute each matrix needed for Quadratic Programming
if strcmp(type_bandwidth,'fixed')
    [yhat_obs,alpha_hat_obs,beta_hat_obs] = ComputeLL_fixed(X,X,y,bandwidth,kernel);
    if grid_point > 0
        [yhat,alpha_hat,beta_hat] = ComputeLL_fixed(X,x,y,bandwidth,kernel);
    else
        yhat = [];
        alpha_hat = 0;
        beta_hat = 0;
    end
    if isempty(X_eval)==0
        [yhat_eval,alpha_hat_eval,beta_hat_eval] = ComputeLL_fixed(X,X_eval,y,bandwidth,kernel);
    else
        yhat_eval = 0;
        beta_hat_eval = 0;
    end
elseif strcmp(type_bandwidth,'variable')
    [yhat_obs,alpha_hat_obs,beta_hat_obs] = ComputeLL_knn(X,X,y,bandwidth,kernel,X_tilde,X_tilde);
    if grid_point > 0
        [yhat,alpha_hat,beta_hat] = ComputeLL_knn(X,x,y,bandwidth,kernel,X_tilde,x_tilde);
    else
        yhat = [];
        alpha_hat = 0;
        beta_hat = 0;
    end
    if isempty(X_eval)==0
        [yhat_eval,alpha_hat_eval,beta_hat_eval] = ComputeLL_knn(X,X_eval,y,bandwidth,kernel,X_tilde,X_eval_tilde);
    else
        yhat_eval = 0;
        beta_hat_eval = 0;
    end
end


end