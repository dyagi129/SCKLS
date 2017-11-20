%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function of Shape Constrained Kernel Least Squares developped
% by Yagi et al (2016).
%
%
% Input arguments:
%       X:              observed input
%       y:              observed 
%       p:              order of polynomial (>=1 (e.g) 1: LC, 2: LQ)
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
function [yhat,alpha_hat,beta_hat,x,perc_const] = SCKLS_p(X,y,p,type_bandwidth,type_grid,kernel,grid_point,bandwidth,x,concavity,increase)


%% Error check
n = size(X,1);
d = size(X,2);


switch nargin,
    case [0,1],
        error('Not enough arguments.')
    case 2,
        p=1;
        type_bandwidth = 'fixed';
        type_grid = 'equal';
        kernel = 'gaussian';
        grid_point = 100;
        bandwidth = std(X)*n^(-1/(4+d)); 
        x =[];
        concavity = 1;
        increase = 1;
    case 3,
        type_bandwidth = 'fixed';
        type_grid = 'equal';
        kernel = 'gaussian';
        grid_point = 100;
        bandwidth = std(X)*n^(-1/(4+d)); 
        x =[];
        concavity = 1;
        increase = 1;
    case 4,
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
    case 5,
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
    case 6,
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        x = [];
        concavity = 1;
        increase = 1;
    case 7,
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
        x = [];
        concavity = 1;
        increase = 1;
    case 8,
        x = [];
        concavity = 1;
        increase = 1;
    case 9,
        concavity = 1;
        increase = 1;
    case 10,
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
    [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_fixed_LP(X,y,bandwidth,kernel,grid_point,x,dx,p); 
elseif strcmp(type_bandwidth,'variable'),
    [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_knn(X,y,bandwidth,kernel,grid_point,X_tilde,x,dx);
end



% %% Drop grid points which do not have any observed points (see Arie (2007))
% if d >= 2 && isempty(Neighbor_indic)==0,
%     [Hmat,fvec,Amat,bvec,row,x] = DropGridWithoutObs(X,Hmat,fvec,Amat,bvec,row,x,x_axis,Neighbor_indic);
% end



%% Solve optimization problem to get SCKLS estimates
violation=1;   
v_pt=[];
            
while violation==1

format short g
%Estimate function and slope
[alpha1,beta1,yhat,Amat,row] = ComputeSCKLS_p(y,X,Hmat,fvec,Amat,bvec,x,v_pt,row,concavity,increase,p);

%Check concavity for estimated points by Afriat inequalities
violation=0;
cnt_violation = 0;

v_pt = [];
cnt = 0;
if concavity == 1,  % Case) convexity
    for i = 1:length(x),
        for j = 1:length(x),
            if(i ~=j)
                if yhat(i) - yhat(j) + 0.001 < beta1(i,:)*(x(i,:) - x(j,:))';                
                    violation=1;
                    cnt=cnt+1;
                    cnt_violation=cnt_violation+1;
                    v_pt(cnt,:)=[i,j];
                end
            end
        end
    end
elseif concavity == 0,  % Case) convexity
    for i = 1:length(x),
        for j = 1:length(x),
            if(i ~=j)
                if yhat(i) - yhat(j) - 0.001 > beta1(i,:)*(x(i,:) - x(j,:))';                
                    violation=1;
                    cnt=cnt+1;
                    cnt_violation=cnt_violation+1;
                    v_pt(cnt,:)=[i,j];
                end
            end
        end
    end
end

if violation == 0
%     fprintf('Satisfied all Afriat inequalities.\n');
else
%     fprintf('Violated %d Afriat inequalities.\n',cnt_violation);
end

end

yhat = yhat;
beta_hat = beta1;
alpha_hat = alpha1;

%Later delete
m=size(x,1);
perc_const = (row + m*d) / (m*(m-1) + m*d);


end