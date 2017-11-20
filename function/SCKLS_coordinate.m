%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function of Shape Constrained Kernel Least Squares developped
% by Yagi et al (2016).
%
%
% Input arguments:
%       X:              observed input
%       y:              observed output
%       type_bandwidth: type of bandwidth ('fixed','variable')
%                       where I use KNN for variable bandwidth
%       type_grid:      type of grid points ('equal','percentile','density')
%       kernel:         type of kernel ('gaussian','uniform','epanechnikov')
%       grid_point:     number of grid points for the estimation
%       bandwidth:      bandwidth (when 'type_bandwidth' = 'variable', put K-value for KNN
%       x:              evaluation points (set this if you want to use specific evaluation points)
%       concavity:      0:convex, 1:concave
%       increase:       0:decreasing, 1:increasing
%
%
% Output arguments:
%       yhat:           functional estimate on each grid points
%       alpha_hat:      intercept estimate on each grid points
%       beta_hat:       slope estimate on each grid points
%       x:              evaluation points
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [yhat,alpha_hat,beta_hat,x] = SCKLS(X,y,type_bandwidth,type_grid,kernel,grid_point,bandwidth)
function [yhat,alpha_hat,beta_hat,x] = SCKLS_coordinate(X,y,type_bandwidth,type_grid,kernel,grid_point,bandwidth,x,concavity,increase)


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
        x =[];
        concavity = 1;
        increase = 1;
    case 3,
        kernel = 'gaussian';
        type_grid = 'equal';
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        x = [];
        concavity = 1;
        increase = 1;
    case 4,
        type_grid = 'equal';
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        x = [];
        concavity = 1;
        increase = 1;
    case 5,
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        x = [];
        concavity = 1;
        increase = 1;
    case 6,
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        x = [];
        concavity = 1;
        increase = 1;
    case 7,
        x = [];
        concavity = 1;
        increase = 1;
    case 8,
        concavity = 1;
        increase = 1;
    case 9,
        increase = 1;
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
if isempty(x),
    % In case that there is no pre-specified grid points, compute evaluation points from observed points.
    [x,dx,x_axis] = grid_x(X,grid_point,type_grid);
else
    dx = [];
    x_axis = [];
end

%% Compute each matrix needed for Quadratic Programming
if strcmp(type_bandwidth,'fixed'),
    [Hmat,fvec,Amat,bvec,lb,ub] = ComputeMatrix_fixed_coordinate_LL2(X,y,bandwidth,kernel,grid_point,x,dx); 
%     [Hmat,fvec,lb,ub] = ComputeMatrix_fixed_coordinate_LP(X,y,bandwidth,kernel,grid_point,x,dx); 
%     [Hmat2,fvec2,Amat2,bvec2,row,Neighbor_indic] = ComputeMatrix_fixed(X,y,bandwidth,kernel,grid_point,x,dx); 
elseif strcmp(type_bandwidth,'variable'),
    [Hmat,fvec,Amat,bvec,row] = ComputeMatrix_knn(X,y,bandwidth,kernel,grid_point,X_tilde,x,dx);
end



%% Drop grid points which do not have any observed points (see Arie (2007))
% if d >= 2,
% %     [Hmat,fvec,Amat,bvec,row,x] = DropGridWithoutObs(X,Hmat,fvec,Amat,bvec,row,x,x_axis,Neighbor_indic);
% end



%% Solve optimization problem to get SCKLS estimates
% if concavity == 1,
%     Amat_temp = Amat; 
% elseif concavity == 0,
%     Amat_temp = -Amat;
% end

options = optimoptions('quadprog','Display','off');
[est] = quadprog(Hmat,fvec,Amat,bvec,[],[],lb,ub,[],options);


% [est] = quadprog(Hmat,fvec,[],[],[],[],-ub,-lb,[],options);
% if increase == 1,
%     [est] = quadprog(Hmat,fvec,[],[],[],[],lb,[],[],options);
% elseif increase == -1,
%     [est] = quadprog(Hmat,fvec,[],[],[],[],[],ub,[],options);
% else
%     [est] = quadprog(Hmat,fvec,Amat_temp,bvec,[],[],[],[],[],options);
% end


%% LL

est = reshape(est,length(x),1+d);
yhat = est(:,1);
beta_hat = est(:,2:(1+d));
alpha_hat = zeros(length(x),1);
for i = 1:length(x),
    alpha_hat(i,:) = yhat(i) - beta_hat(i,:)*(x(i,:)).';
end


% %% LP
% Num_FP = d;
% Num_SP = 1/2 * d * (d+1);
% 
% est = reshape(est,length(x),1+Num_FP+Num_SP);
% yhat = est(:,1);
% beta_hat = est(:,2:(1+Num_FP));
% beta2_hat = est(:,(1+Num_FP+1):end);
% alpha_hat = zeros(n,1);
% for i = 1:length(x),
%     alpha_hat(i,:) = yhat(i) - beta_hat(i,:)*(x(i,:)).';
% end
% 
% % figure()
% % hold on
% % plot3(X(:,1),X(:,2),y,'.')
% % plot3(x(:,1),x(:,2),yhat,'.')

%Later delete
m=size(x,1);


end