%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to compute matrix for quadratic programming
% (H,f,A,b)
% min  1/2*x'*H*x + f'*x
% s.t. A*x <= b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_fixed_coordinate(X,y,h,kernel,grid_point,x,dx)



%% Initial check
n = size(y,1);
m = size(x,1);
% size returns a vector of length 2 with height and width of x
d = size(X,2); 
% Reassign d with the value of d(2) makes d a scalar with the value indicating the 
% number of input variables


%Equally spaced grid.
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

%Small x is point for evaluation
%Capital X is observed points

l = zeros(length(x),d);



%% Define kernel function
if strcmp(kernel,'gaussian'),
    %Gaussian kernel
    kerf = @(z) exp(-z.*z/2)/sqrt(2*pi);

elseif strcmp(kernel,'uniform'),
    %Uniform kernel
    kerf = @(z) 1/2*(abs(z)<=1);

elseif strcmp(kernel,'epanechnikov'),
    %Epanechnikov kernel
    kerf = @(z) 3/4*(1-z^2)*(abs(z)<=1);
end




%Calculate kernel weights before declaring objective function
prod_k = ones(length(x),n);

for i = 1:length(x),
    for j = 1:n,
        for k = 1:d,    %Daisuke: loop to calculate product kernel
            prod_k(i,j) = prod_k(i,j)*kerf((X(j,k)-x(i,k))/h(k));
        end
    end
end



%% Construct matrix for quadprog
cnt_obj = 0;
cnt_cnst = 0;

sum_prod_k = zeros(length(x),1);
sum_y_times_k=zeros(length(x),1);
sum_diff_X_times_yk=zeros(length(x),d);
sum_diff_X_times_k=zeros(length(x),d);
sum_diff_Xsq_times_k=zeros(length(x),d);
sum_diff_X_times_diff_X=zeros(length(x),d,d);


sum_prod_k = sum(prod_k,2);
sum_y_times_k = prod_k * y;
for i = 1:length(x),
    for k = 1:d,
        sum_diff_X_times_yk(i,k) = sum((X(:,k)-x(i,k)) .*y .*prod_k(i,:)');
        sum_diff_X_times_k(i,k) = sum((X(:,k)-x(i,k)) .* prod_k(i,:)');
        sum_diff_Xsq_times_k(i,k) = sum((X(:,k)-x(i,k)).^2 .* prod_k(i,:)');
        for l = 1:d,
             sum_diff_X_times_diff_X(i,k,l) = sum((X(:,k)-x(i,k)) .* (X(:,l)-x(i,l)) .* prod_k(i,:)');
        end
    end
end


Hmat = zeros(length(x)+length(x)*d);
for i = 1:length(x)
    for j = 0:d
        for k = 0:d
            if j==0 && k==0 
                Hmat(i,i) = sum_prod_k(i);
            elseif j==0 && k~=0 
                Hmat(i+j*length(x),i+k*length(x)) = sum_diff_X_times_k(i,k);
            elseif j~=0 && k==0
                Hmat(i+j*length(x),i+k*length(x)) = sum_diff_X_times_k(i,j);
            elseif j~=0 && k~=0 && j==k
                Hmat(i+j*length(x),i+k*length(x)) = sum_diff_Xsq_times_k(i,j);
            elseif j~=0 && k~=0 && j~=k
                Hmat(i+j*length(x),i+k*length(x)) = sum_diff_X_times_diff_X(i,j,k);
            end
        end
    end
end

fvec_alpha = sum_y_times_k;
fvec_beta = reshape(sum_diff_X_times_yk,length(x)*d,1);
fvec = -[fvec_alpha;fvec_beta];

% Amat_afriat = [];



if isempty(dx)==0,
    % consider only neibors and violated points in the previous loop
    % resize x to make it x_ij
    if d == 2,
        for ii = 1:d,
            x_ij(ii,:,:) = reshape(x(:,ii),Num_grid,Num_grid);
        end
    elseif d == 1,
        for ii = 1:d,
            x_ij(ii,:) = reshape(x(:,ii),Num_grid,1);
        end
    elseif d == 3,
        for ii = 1:d,
            x_ij(ii,:,:,:) = reshape(x(:,ii),Num_grid,Num_grid,Num_grid);
        end
    elseif d == 4,
        for ii = 1:d,
            x_ij(ii,:,:,:,:) = reshape(x(:,ii),Num_grid,Num_grid,Num_grid,Num_grid);
        end
    elseif d == 4,
        for ii = 1:d,
            x_ij(ii,:,:,:,:,:) = reshape(x(:,ii),Num_grid,Num_grid,Num_grid,Num_grid,Num_grid);
        end       
    end

    x_index = repmat([1:1:Num_grid]',1,d);
    if d == 2,
        [x1,x2]=ndgrid(x_index(:,1),x_index(:,2));
        x_index=[x1(:),x2(:)];
    elseif d == 1,
        [x1]=ndgrid(x_index(:,1));
        x_index=[x1(:)];
    elseif d == 3,
        [x1,x2,x3]=ndgrid(x_index(:,1),x_index(:,2),x_index(:,3));
        x_index=[x1(:),x2(:),x3(:)];  
    elseif d == 4,
        [x1,x2,x3,x4]=ndgrid(x_index(:,1),x_index(:,2),x_index(:,3),x_index(:,4));
        x_index=[x1(:),x2(:),x3(:),x4(:)];  
    elseif d == 5,
        [x1,x2,x3,x4,x5]=ndgrid(x_index(:,1),x_index(:,2),x_index(:,3),x_index(:,4),x_index(:,5));
        x_index=[x1(:),x2(:),x3(:),x4(:),x5(:)];  
    end

    Neighbor_indic = zeros(length(x));

    for i = 1:length(x),
        for j = 1:length(x),
            if i ~= j,
                flag = 0;
                for k = 1:d,
                    % check neighbor 
                       
                    if d == 2,
                        if x_ij(k,x_index(i,1),x_index(i,2)) - x_ij(k,x_index(j,1),x_index(j,2)) > 0,

                            if abs(x_ij(k,x_index(i,1),x_index(i,2)) - x_ij(k,x_index(j,1),x_index(j,2))) <= dx(x_index(i,k)-1,k),
                                flag = flag + 1;
                            end

                        elseif x_ij(k,x_index(i,1),x_index(i,2)) - x_ij(k,x_index(j,1),x_index(j,2)) < 0,

                            if abs(x_ij(k,x_index(i,1),x_index(i,2)) - x_ij(k,x_index(j,1),x_index(j,2))) <= dx(x_index(i,k),k),
                                flag = flag + 1;
                            end

                        elseif x_ij(k,x_index(i,1),x_index(i,2)) - x_ij(k,x_index(j,1),x_index(j,2)) == 0,

                            flag = flag + 1;

                        end
                    
                    % Just do the same thing for each input dimension
                    elseif d == 1,
                        if x_ij(k,x_index(i,1)) - x_ij(k,x_index(j,1)) > 0,
                            if abs(x_ij(k,x_index(i,1)) - x_ij(k,x_index(j,1))) <= dx(x_index(i,k)-1,k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1)) - x_ij(k,x_index(j,1)) < 0,
                            if abs(x_ij(k,x_index(i,1)) - x_ij(k,x_index(j,1))) <= dx(x_index(i,k),k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1)) - x_ij(k,x_index(j,1)) == 0,
                            flag = flag + 1;
                        end
                    elseif d == 3,
                        if x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3)) > 0,
                            if abs(x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3))) <= dx(x_index(i,k)-1,k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3)) < 0,
                            if abs(x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3))) <= dx(x_index(i,k),k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3)) == 0,
                            flag = flag + 1;
                        end
                    elseif d == 4,
                        if x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4)) > 0,
                            if abs(x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4))) <= dx(x_index(i,k)-1,k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4)) < 0,
                            if abs(x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4))) <= dx(x_index(i,k),k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4)) == 0,
                            flag = flag + 1;
                        end
                    elseif d ==5,
                        if x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4),x_index(i,5)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4),x_index(j,5)) > 0,
                            if abs(x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4),x_index(i,5)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4),x_index(j,5))) <= dx(x_index(i,k)-1,k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4),x_index(i,5)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4),x_index(j,5)) < 0,
                            if abs(x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4),x_index(i,5)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4),x_index(j,5))) <= dx(x_index(i,k),k),
                                flag = flag + 1;
                            end
                        elseif x_ij(k,x_index(i,1),x_index(i,2),x_index(i,3),x_index(i,4),x_index(i,5)) - x_ij(k,x_index(j,1),x_index(j,2),x_index(j,3),x_index(j,4),x_index(j,5)) == 0,
                            flag = flag + 1;
                        end
                    end
                    
                    
                    
                    
                end
                if flag == d,
                    Neighbor_indic(i,j) = 1;
                end
            end
        end
    end



    [row_index,col_index] = find(Neighbor_indic==1);
    row = 1;
    Amat_afriat = sparse(length(row_index), length(x)*(d+1));

    for i = 1:length(row_index)
        Amat_afriat(row,row_index(i)) = -1;
        Amat_afriat(row,col_index(i)) = 1;
        for k = 1:d
            Amat_afriat(row,row_index(i) + length(x)*k) = x(row_index(i),k)-x(col_index(i),k);
        end
        row = row + 1;
    end

else
    % Put all combination of afriat inequalities (obs points)
    row = 1;
    Amat_afriat = sparse(m*(m-1), length(x)*(d+1));
    Neighbor_indic = [];
    for ii = 1:m,
        for jj = 1:m,
            if ii~=jj,
                Amat_afriat(row,ii) = -1;
                Amat_afriat(row,jj) = 1;
                for k = 1:d
                    Amat_afriat(row,ii + m*k) = x(ii,k)-x(jj,k);
                end
                row = row + 1;

            end
        end
    end
end



Amat = Amat_afriat;
bvec=zeros(row-1,1);




end