%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to drop the grid points 
% which do not have observation points in neighbors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Hmat,fvec,Amat,bvec,row,x] = DropGridWithoutObs(X,Hmat,fvec,Amat,bvec,row,x,x_axis,Neighbor_indic)



%% Some values for later calculation
% Number of observations
n = size(X,1);

% Number of inputs
d = size(X,2); 

dlt_list_index = [];
cnt = 1;
Num_grid = round(length(x)^(1/d));

% Type 1: delete all grid points which do not have obs in neighbors
% Type 2: same as type 1 but without hole in grid points space
type = 1;



%% Type 1. Delete all grid points which do not have obs points in neighbors

% Check each grid points has observation points between neighbors
for ii = 1:length(x)
    neighbor_index_tmp = find(Neighbor_indic(ii,:)==1)';
    
    % Define set of grid points x itself and neighbors
    set_x_neighbor = [x(ii,:);x(neighbor_index_tmp,:)];
    
    % Calculate maximum and minimum of the set of x
    max_x = max(set_x_neighbor);
    min_x = min(set_x_neighbor);
    
    temp_flag_exist = 0;
    for jj = 1:n
        % If obs points are existing between neighbors, flag = 1 (exist!)
        if X(jj,:) >= min_x & X(jj,:) <= max_x,
            temp_flag_exist = 1;
            break
        end
    end
    
    if temp_flag_exist == 0,
        % Delete grid points which don't have observed points (store index)
        dlt_list_index(cnt,1) = ii;
        cnt = cnt + 1;
    end
end



%% Type 2. Delete grid points which do not have obs points in neighbors, but without hole in grid points
if type == 2,
    % For each axis, check whether there is hole or not
    for kk = 1:d,
        for ii = 1:Num_grid^(d-1),
            
            % Calculate the all combination of x_axis (need in case of d>2)
            tmp_ind = find(1:d~=kk);
            if d == 2,
                x_axis_tmp=[x_axis(tmp_ind(1)),x_axis(tmp_ind(2))];
            elseif d == 3,
                [x1,x2]=ndgrid(x_axis(:,tmp_ind(1)),x_axis(:,tmp_ind(2)));
                x_axis_tmp=[x1(:),x2(:)];  
            elseif d == 4,
                [x1,x2,x3]=ndgrid(x_axis(:,tmp_ind(1)),x_axis(:,tmp_ind(2)),x_axis(:,tmp_ind(3)));
                x_axis_tmp=[x1(:),x2(:),x3(:)];  
            elseif d == 5,
                [x1,x2,x3,x4]=ndgrid(x_axis(:,tmp_ind(1)),x_axis(:,tmp_ind(2)),x_axis(:,tmp_ind(3)),x_axis(:,tmp_ind(4)));
                x_axis_tmp=[x1(:),x2(:),x3(:),x4(:)];  
            end
            
            % Define axis and obtain all grid points which on the axis
            list_x = find(ismember(x(:,1:end~=kk),x_axis_tmp(ii,:),'rows'));%find(x(:,1:end~=kk)==x_axis(ii,1:end~=kk));
            
            % Obtain list of grid points which still not deleted
            list_x_still_exist = list_x(~ismember(list_x,dlt_list_index));
            
            % Define the list of grid points which shouldn't be deleted
            list_stop_dlt = [min(list_x_still_exist):Num_grid^(kk-1):max(list_x_still_exist)]';
            list_stop_dlt = list_stop_dlt(~ismember(list_stop_dlt,list_x_still_exist));
            if length(list_stop_dlt)>0,
               aaa=1; 
            end
            % Stop delete by deleting axis
            dlt_list_index(ismember(dlt_list_index,list_stop_dlt)) = [];
        end
    end
end
    
    
    
%% Delete grid points from objective function, constraints and definition of grid points

[index_Amat_dlt,tmp] = find(Amat(:,dlt_list_index)==1|Amat(:,dlt_list_index)==-1);
index_Amat_dlt = unique(index_Amat_dlt);
Amat(index_Amat_dlt,:) = [];
bvec(index_Amat_dlt,:) = [];

for kk = d:-1:0
    Hmat(dlt_list_index + kk*length(x),:) = [];
    Hmat(:,dlt_list_index + kk*length(x)) = [];
    fvec(dlt_list_index + kk*length(x)) = [];
    Amat(:,dlt_list_index + kk*length(x))=[];
end

x(dlt_list_index,:) = [];

% Update row
row = size(Amat,1) + 1;



%% Graph output (only for testing reason)
% figure()
% axis tight, hold on
% % plot(X(:,1),X(:,2),'.')
% % plot(x(:,1),x(:,2),'.')
% plot3(X(:,1),X(:,2),X(:,3),'.')
% plot3(x(:,1),x(:,2),x(:,3),'.')


end