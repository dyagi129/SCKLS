%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function to get the grid points for SCKLS/CWB.
%
%
%
% Input arguments:
%       X:              observed input
%       grid_point:     number of grid points
%       type_grid:      type of grid you want to use
%                       'equal':        equally spaced grid
%                       'percentile':   use percentile of observed points
%                       'density':      use kernel density estimation of observed points
%
%
% Output arguments:
%       x:              grid points
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [x,dx,x_axis] = grid_x(X,grid_point,type_grid)

%% Error check
switch nargin,
    case [0,1],
        error('Not enough arguments.')
    case 2,
        type_grid = 'equal'
end

if strcmp(type_grid,'equal') + strcmp(type_grid,'percentile') + strcmp(type_grid,'density') == 0,
    error('"type_grid" has wrong name.')
end




%% Get grid points

n = size(X,1);
d = size(X,2); 

% Number of grid for each dimension
Num_grid = round(grid_point^(1/d));

x=zeros(Num_grid,d);
if strcmp(type_grid,'equal') == 1,
    % Equally spaced grid
    for i = 1:d,
        x(:,i) = linspace(min(X(:,i))', max(X(:,i))', Num_grid);
    end
    dx = diff(x)*1.001;
elseif strcmp(type_grid,'percentile') == 1,
    % Equally percentile of observed data 
    perc_list = linspace(0,100,Num_grid);
    for i = 1:d,
        x(:,i) = prctile(X(:,i),perc_list);
    end
    dx = diff(x)*1.001;
elseif strcmp(type_grid,'density') == 1,
    % Equally density of observed data (kernel density estimation)
    
    for i = 1:d,
        min_list = ksdensity(X(:,i),min(X(:,i)),'function','cdf');
        max_list = ksdensity(X(:,i),max(X(:,i)),'function','cdf');
        cdf_list = linspace(min_list,max_list,Num_grid);
%         min_list = ksdensity(X(:,i),0,'function','cdf');
%         cdf_list = linspace(min_list,0.99,Num_grid);
        x(:,i) = ksdensity(X(:,i),cdf_list,'function','icdf');
    end
    dx = diff(x)*1.001;
end

x_axis = x;

if d == 2,
    [x1,x2]=ndgrid(x(:,1),x(:,2));
    x=[x1(:),x2(:)];
elseif d == 3,
    [x1,x2,x3]=ndgrid(x(:,1),x(:,2),x(:,3));
    x=[x1(:),x2(:),x3(:)];  
elseif d == 4,
    [x1,x2,x3,x4]=ndgrid(x(:,1),x(:,2),x(:,3),x(:,4));
    x=[x1(:),x2(:),x3(:),x4(:)];  
elseif d == 5,
    [x1,x2,x3,x4,x5]=ndgrid(x(:,1),x(:,2),x(:,3),x(:,4),x(:,5));
    x=[x1(:),x2(:),x3(:),x4(:),x5(:)];  
end

% figure()
% % axis tight, hold on
% plot(x(:,1),x(:,2),'.');
% plot(X(:,1),X(:,2),'. ');

end
