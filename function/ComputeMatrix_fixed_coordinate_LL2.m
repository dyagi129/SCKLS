%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to compute matrix for quadratic programming
% (H,f,A,b)
% min  1/2*x'*H*x + f'*x
% s.t. A*x <= b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Hmat,fvec,Amat,bvec,lb,ub] = ComputeMatrix_fixed_coordinate_LL(X,y,h,kernel,grid_point,x,dx)



%% Initial check
n = size(y,1);
m = size(x,1);
% size returns a vector of length 2 with height and width of x
d = size(X,2); 
% Reassign d with the value of d(2) makes d a scalar with the value indicating the 
% number of input variables


%Equally spaced grid.
Num_grid = round(grid_point^(1/d));

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


%% Coordinate-wise concavity

Num_Concavity = (Num_grid-2) * Num_grid^(d-1) * d;
Amat_afriat = zeros(Num_Concavity, length(x)*(d+1));

unique_x = zeros(Num_grid,d);
for kk = 1:d
    unique_x(:,kk) = sort(unique(x(:,kk)));
end

row = 1;
for kk = 1:d
    rest_x = x;
    rest_x(:,kk) = [];
    unique_rest_x = unique(rest_x,'rows');
    for ii = 1:length(unique_rest_x)

        for jj = 1:length(unique_x(:,kk))-2
            ind_tmp1 = find(x(:,kk) == unique_x(jj,kk));
            ind_tmp2 = find(ismember(rest_x, unique_rest_x(ii,:), 'rows'));
            ind_1 = intersect(ind_tmp1,ind_tmp2);
            
            ind_tmp1 = find(x(:,kk) == unique_x(jj+1,kk));
            ind_2 = intersect(ind_tmp1,ind_tmp2);
            
            ind_tmp1 = find(x(:,kk) == unique_x(jj+2,kk));
            ind_3 = intersect(ind_tmp1,ind_tmp2);
            
            Amat_afriat(row,ind_1) = 1/(x(ind_2,kk)-x(ind_1,kk));
            Amat_afriat(row,ind_2) = -1/(x(ind_3,kk)-x(ind_2,kk)) - 1/(x(ind_2,kk)-x(ind_1,kk));
            Amat_afriat(row,ind_3) = 1/(x(ind_3,kk)-x(ind_2,kk));
            row = row+1;
            
            % First slope should be positive
            if jj == 1
                Amat_afriat(row,ind_1) = 1/(x(ind_2,kk)-x(ind_1,kk));
                Amat_afriat(row,ind_2) = -1/(x(ind_2,kk)-x(ind_1,kk));
                row = row+1;
            % Last slope should be positive
            elseif jj == length(unique_x(:,kk))-2
                Amat_afriat(row,ind_2) = 1/(x(ind_3,kk)-x(ind_2,kk));
                Amat_afriat(row,ind_3) = -1/(x(ind_3,kk)-x(ind_2,kk));
                row = row+1;
            end
        end
        
        
        row = row+1;
    end
end



Amat = Amat_afriat;
bvec=zeros(size(Amat,1),1);

lb = [-inf(m,1);-inf(m*d,1)];
ub = [inf(m,1);inf(m*d,1)];


end