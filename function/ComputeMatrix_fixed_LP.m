%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to compute matrix for quadratic programming
% (H,f,A,b)
% min  1/2*x'*H*x + f'*x
% s.t. A*x <= b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_fixed_LP(X,y,h,kernel,grid_point,x,dx,p)



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

num_estimates = 0;
for l = 0:p 
    num_estimates = num_estimates + factorial(d+l-1)/(factorial(d-1)*factorial(l));
end
Hmat = zeros(num_estimates*length(x));
fvec = zeros(num_estimates*length(x),1);

for i = 1:length(x)   
    
    W = zeros(n, num_estimates);
    W(:,1) = ones(n,1);
    curr_col = 2;
    for l = 1:p
        comb = nmultichoosek(1:d,l);
        for ll = 1:length(comb)
            W(:,curr_col) = prod( X(:,comb(ll,:)) - repmat(x(i,comb(ll,:)),n,1), 2);
            curr_col = curr_col + 1;
        end
    end
    
    A = W' * diag(prod_k(i,:)) * W;
    B = W' * diag(prod_k(i,:)) * y;
    
    for l = 1:num_estimates
        for m = 1:num_estimates
            Hmat( i+ (l-1)*length(x), i+ (m-1)*length(x)) = A(l,m);
        end
        fvec( i+ (l-1)*length(x)) = -B(l);
    end
    
end    



%% Afriat inequalities
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
    Amat_afriat = sparse(length(row_index), length(x)*num_estimates);

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
    Amat_afriat = sparse(m*(m-1), length(x)*(num_estimates));
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