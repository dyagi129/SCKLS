%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function for calculating lower bound of bandwidth
%
% This cod is just calculating the largest distance from grid points to
% observed points. Then, I defined lower bound of bandwidth based on
% pre-defined %.
%
%
% Input arguments:
%       X:              observed input
%       y:              observed output
%       h:              optimal bandwidth obtained before
%       grid_point:     # of grid points
%       per_distance:   % of distance for defining lower bound   
%
% Output arguments:
%       h_lb:           lower bound of bandwidth
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [h] = LowerBoundBandwidth(X,y,x,h,grid_point,per_distance)



%% Initial check
n = size(y,1);
d = size(X,2); 
l = zeros(n,d);



switch nargin,
    case [0,1,2],
        error('Not enough arguments.')
    case 3,
        per_distance = 0.5;
end

%Normalize X
X = X./repmat(std(X),n,1);



%% Define Equally spaced grid
Num_grid = round(grid_point^(1/d));

% for i = 1:d,
%     x(:,i) = linspace(min(X(:,i))', max(X(:,i))', Num_grid);
%     %Calculate distance of equally space grid
%     dx(i) = (x(2,i)-x(1,i))*1.1;
% end
% 
% 
% if d == 2,
%     [x1,x2]=ndgrid(x(:,1),x(:,2));
%     x=[x1(:),x2(:)];
% elseif d == 3,
%     [x1,x2,x3]=ndgrid(x(:,1),x(:,2),x(:,3));
%     x=[x1(:),x2(:),x3(:)];  
% elseif d == 4,
%     [x1,x2,x3,x4]=ndgrid(x(:,1),x(:,2),x(:,3),x(:,4));
%     x=[x1(:),x2(:),x3(:),x4(:)];  
% elseif d == 5,
%     [x1,x2,x3,x4,x5]=ndgrid(x(:,1),x(:,2),x(:,3),x(:,4),x(:,5));
%     x=[x1(:),x2(:),x3(:),x4(:),x5(:)];  
% end

% type_grid = 'density';
% [x,dx] = grid_x(X,grid_point,type_grid);

%Small x is point for evaluation
%Capital X is observed points

m = size(x,1);



%% Calculate distance from grid to obs


Dist = zeros(length(x),n);
for kk = 1:d,
    Dist = Dist + (repmat(X(:,kk)',length(x),1) - repmat(x(:,kk),1,n)).^2;
end
Dist = sqrt(Dist);

%closest obs points for each grid points (1 by m vector)
[min_val, obs_index] = min(Dist,[],2); 

%calculating the longest distance for each dimension
max_distance = max(abs(X(obs_index,:) - x)); 

h_lb = max_distance .* per_distance .* std(X);

h(abs(h) <= abs(h_lb)) = h_lb(abs(h) <= abs(h_lb));
h_rev_display = sprintf(' %.5g ',h);
fprintf('Revised bandwidth: [%s]\n',h_rev_display);



end