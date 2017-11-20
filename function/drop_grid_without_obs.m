%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to drop the grid points 
% which do not have observation points in neighbors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [x,dlt_index] = drop_grid_without_obs(x,X)



%% Some values for later calculation
% Number of observations
n = size(X,1);

% Number of inputs
d = size(X,2); 

% Num grid
Num_grid = round(length(x)^(1/d));
m = size(x,1);

cnt = 1;

%% Select type
% Type 1: Grid should have observation inside the neighborhoods 
% Type 2: Grid is inside the convex hull of observed points 
Type = 2;

%% Type 1. Delete all grid points which do not have obs points in neighbors

if Type == 1
    % Check each grid points has observation points between neighbors
    for ii = 1:m
        distance_grid = (repmat(x(ii,:),m,1)-x).^2;

        min_distance = zeros(1,d);
        for kk = 1:d
            tmp = sort(unique(distance_grid(:,kk)));
            min_distance(kk) = tmp(2);    
        end
        neighbor_index_tmp = find(sum(distance_grid<=repmat(min_distance*1.1,m,1),2)==d)';

        % Define set of grid points x itself and neighbors
        set_x_neighbor = [x(neighbor_index_tmp,:)];

        % Calculate maximum and minimum of the set of x
        max_x = max(set_x_neighbor);
        min_x = min(set_x_neighbor);

        temp_flag_exist = 0;
        for jj = 1:n
            % If obs points are existing between neighbors, flag = 1 (exist!)
            if sum(X(jj,:) >= min_x)==d && sum(X(jj,:) <= max_x)==d
                temp_flag_exist = 1;
                break
            end
        end

        if temp_flag_exist == 0,
            % Delete grid points which don't have observed points (store index)
            dlt_index(cnt,1) = ii;
            cnt = cnt + 1;
        end
    end
end



%% Type 2
if Type == 2
    
    % Compute convex hull
    index_CH = convhull(X);
    
    % Check whether grid points is inside convex hull or not
    IN = inpolygon(x(:,1),x(:,2),X(index_CH,1),X(index_CH,2));
    dlt_index = [1:m]';
    dlt_index(IN) = [];
end



%% Delete grid points from objective function, constraints and definition of grid points
x(dlt_index,:) = [];
% index_grid = [1:m]';
% index_grid(dlt_list_index) = [];


%% Graph output (only for testing reason)
% figure()
% axis tight, hold on
% plot(X(:,1),X(:,2),'.')
% plot(x(:,1),x(:,2),'.')
% plot3(X(:,1),X(:,2),X(:,3),'.')
% plot3(x(:,1),x(:,2),x(:,3),'.')


end