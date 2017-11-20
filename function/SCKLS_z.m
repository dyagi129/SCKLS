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
%
%
% Output arguments:
%       ahat:           functional estimate on each grid points
%       bhat:           slope estimate on each grid points
%       x:              equally spaced grid points
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [yhat,alpha_hat,beta_hat,x] = SCKLS(X,y,type_bandwidth,type_grid,kernel,grid_point,bandwidth)
function [yhat,alpha_hat,beta_hat,delta_hat,CI,p_value,x,perc_const] = SCKLS_z(X,y,Z,type_bandwidth,type_grid,kernel,grid_point,bandwidth)

addpath('C:\Share\Google Drive\workspace\function\LC')

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
    case 3,
        kernel = 'gaussian';
        type_grid = 'equal';
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
    case 4,
        type_grid = 'equal';
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
    case 5,
        grid_point = 100;
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
    case 6,
        if strcmp(type_bandwidth,'fixed'),
            bandwidth = std(X)*n^(-1/(4+d)); 
        elseif strcmp(type_bandwidth,'variable'),
            bandwidth = round(n/10);
        end
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
if grid_point == 0,
    % When grid point = 0, use observation points as evaluation points
    x = X;
    dx = [];
else
    [x,dx,x_axis] = grid_x(X,grid_point,type_grid);
end

%% Compute each matrix needed for Quadratic Programming
if strcmp(type_bandwidth,'fixed'),
    [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_fixed_z(X,y,Z,bandwidth,kernel,grid_point,x,dx); 
elseif strcmp(type_bandwidth,'variable'),
    [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_knn_z(X,y,Z,bandwidth,kernel,grid_point,X_tilde,x,dx);
end



%% Drop grid points which do not have any observed points (see Arie (2007))
% if d >= 2,
%     [Hmat,fvec,Amat,bvec,row,x] = DropGridWithoutObs(X,Hmat,fvec,Amat,bvec,row,x,x_axis,Neighbor_indic);
% end


%% Solve optimization problem to get SCKLS estimates
violation=1;   
v_pt=[];
            
while violation==1

format short g
%Estimate function and slope
[alpha1,beta1,delta1,yhat,Amat,row] = ComputeSCKLS_z(y,X,Z,Hmat,fvec,Amat,bvec,x,v_pt,row);

%Check concavity for estimated points by Afriat inequalities
violation=0;
cnt_violation = 0;

v_pt = [];
cnt = 0;
for i = 1:length(x),
    for j = 1:length(x),
        if(i ~=j)
            if yhat(i) - yhat(j) + mean(y)*0.001 < beta1(i,:)*(x(i,:) - x(j,:))';                
                violation=1;
                cnt=cnt+1;
                cnt_violation=cnt_violation+1;
                v_pt(cnt,:)=[i,j];
            end
        end
    end
end

if violation == 0
    fprintf('Satisfied all Afriat inequalities.\n');
else
    fprintf('Violated %d Afriat inequalities.\n',cnt_violation);
end

end


yhat = yhat;
beta_hat = beta1;
alpha_hat = alpha1;
delta_hat = delta1;


yhat_obs = zeros(n,1);

for jj = 1:size(X,1)
    [yhat_obs(jj,1)] = min(alpha_hat + sum(beta_hat.*repmat(X(jj,:),length(beta_hat),1),2));
end
y_tilde = y - yhat_obs;
Z_tilde = Z;

delta_hat_tmp = (Z_tilde' * Z_tilde)^(-1) * Z_tilde' * y_tilde;

% Compute confidence interval/p-value
VarCoeff = sum((y_tilde - Z_tilde*delta_hat).^2)/(n-size(X,2))*(Z_tilde'*Z_tilde)^(-1);
for kk = 1:size(Z,2),
   SE(kk,1) = sqrt(VarCoeff(kk,kk)); 
end
t_stat = [delta_hat./SE];
significant_level = 0.05;
alphaup = 1-significant_level/2;
alphalow = significant_level/2;
upp = tinv(alphaup,n-size(X,2));
low = tinv(alphalow,n-size(X,2)); 
CI = [delta_hat + low*SE, delta_hat + upp*SE];
p_value = tcdf(-abs(t_stat),n-size(X,2))*2;





%Later delete
m=size(x,1);
perc_const = (row + m*d) / (m*(m-1) + m*d);


end