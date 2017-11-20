%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to compute matrix for quadratic programming
% (H,f,A,b)
% min  1/2*x'*H*x + f'*x
% s.t. A*x <= b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Hmat,fvec,Amat,bvec,row,Neighbor_indic] = ComputeMatrix_fixed_z(X,y,Z,h,kernel,grid_point,x,dx)



%% Initial check
n = size(y,1);
m = size(x,1);
% size returns a vector of length 2 with height and width of x
d = size(X,2); 

d2 = size(Z,2);
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


K = zeros(m,1);
K_dX2 = zeros(m,d,d);
K_Z2 = zeros(m,d2,d2);

K_dX = zeros(m,d);
K_Z = zeros(m,d2);
K_dX_Z = zeros(m,d,d2);

K_y = zeros(m,1);
K_y_dX = zeros(m,d);
K_y_Z = zeros(m,d2);

for i = 1:length(x),
    K(i,1) = sum(prod_k(i,:));
    K_y(i,1) = sum(y .* prod_k(i,:)');
    for k = 1:d,
        K_dX(i,k) = sum((X(:,k)-x(i,k)) .* prod_k(i,:)');
        K_y_dX(i,k) = sum((X(:,k)-x(i,k)) .* y .* prod_k(i,:)');
        for l = 1:d,
            K_dX2(i,k,l) = sum((X(:,k)-x(i,k)) .* (X(:,l)-x(i,l)) .* prod_k(i,:)');
        end
        for l = 1:d2,
            K_dX_Z(i,k,l) = sum((X(:,k)-x(i,k)) .* Z(:,l) .* prod_k(i,:)'); 
        end
    end
    
    for k = 1:d2,
        K_Z(i,k) = sum(Z(:,k).* prod_k(i,:)');
        K_y_Z(i,k) = sum(Z(:,k) .* y .* prod_k(i,:)');
        for l = 1:d2,
            K_Z2(i,k,l) = sum(Z(:,k) .* Z(:,l) .* prod_k(i,:)');
        end
    end
end


Hmat = zeros(m*(d+1)+d2);
fvec = zeros(m*(d+1)+d2,1);

for i = 1:m,
    Hmat(i,i) =  K(i);
    for k = 1:d,
        Hmat(i,m+(k-1)*m+i) = K_dX(i,k);
        Hmat(m+(k-1)*m+i,i) = K_dX(i,k);
        for l = 1:d,
            Hmat(m+(k-1)*m+i,m+(l-1)*m+i) = K_dX2(i,k,l);
        end
    end
    
    for k = 1:d2,
        Hmat(i,(d+1)*m+k) = K_Z(i,k);
        Hmat((d+1)*m+k,i) = K_Z(i,k);
        for l = 1:d2,
            Hmat((d+1)*m+k,(d+1)*m+l) =  sum(K_Z2(:,k,l));
        end
    end
    
    for k = 1:d,
        for l = 1:d2,
            Hmat(m+(k-1)*m+i,(d+1)*m+l) = K_dX_Z(i,k,l);
            Hmat((d+1)*m+l,m+(k-1)*m+i) = K_dX_Z(i,k,l);
        end
    end   
end

fvec = -[K_y;reshape(K_y_dX,m*d,1);sum(K_y_Z,1)];







% sum_prod_k = zeros(length(x),1);
% sum_y_times_k=zeros(length(x),1);
% sum_diff_X_times_yk=zeros(length(x),d);
% sum_diff_X_times_k=zeros(length(x),d);
% sum_diff_Xsq_times_k=zeros(length(x),d);
% sum_diff_X_times_diff_X=zeros(length(x),d,d);
% 
% sum_Z_times_k = prod_k * Z;
% sum_Zsq_times_k = zeros(d2,d2,length(x));
% sum_diff_X_times_Z = zeros(length(x),d,d2);
% sum_y_times_Zk = zeros(d2,1);
% for k = 1:d2,
%     sum_y_times_Zk(k,1) = sum(prod_k * (y .* Z(:,k)));
% end
% 
% sum_prod_k = sum(prod_k,2);
% sum_y_times_k = prod_k * y;
% for i = 1:length(x),
%     for k = 1:d,
%         sum_diff_X_times_yk(i,k) = sum((X(:,k)-x(i,k)) .*y .*prod_k(i,:)');
%         sum_diff_X_times_k(i,k) = sum((X(:,k)-x(i,k)) .* prod_k(i,:)');
%         sum_diff_Xsq_times_k(i,k) = sum((X(:,k)-x(i,k)).^2 .* prod_k(i,:)');
%         for l = 1:d,
%              sum_diff_X_times_diff_X(i,k,l) = sum((X(:,k)-x(i,k)) .* (X(:,l)-x(i,l)) .* prod_k(i,:)');
%         end
%         for l = 1:d2,
%             sum_diff_X_times_Z(i,k,l) = sum((X(:,k)-x(i,k)) .* Z(:,l) .* prod_k(i,:)');
%         end
%     end
% end
% 
% 
% for k = 1:d2,
%     for l = 1:d2,
%         for i = 1:length(x),
%              sum_Zsq_times_k(k,l,i) = sum(Z(:,k) .* Z(:,l) .* prod_k(i,:)');
%         end
%     end
% end
% sum_Zsq_times_k = sum(sum_Zsq_times_k,3);
% 
% 
% 
% Hmat = zeros(length(x)+length(x)*d+d2);
% for i = 1:length(x)
%     for j = 0:d
%         for k = 0:d
%             if j==0 && k==0 
%                 Hmat(i,i) = sum_prod_k(i);
%             elseif j==0 && k~=0 
%                 Hmat(i+j*length(x),i+k*length(x)) = sum_diff_X_times_k(i,k);
%             elseif j~=0 && k==0
%                 Hmat(i+j*length(x),i+k*length(x)) = sum_diff_X_times_k(i,j);
%             elseif j~=0 && k~=0 && j==k
%                 Hmat(i+j*length(x),i+k*length(x)) = sum_diff_Xsq_times_k(i,j);
%             elseif j~=0 && k~=0 && j~=k
%                 Hmat(i+j*length(x),i+k*length(x)) = sum_diff_X_times_diff_X(i,j,k);
%             end
%         end
%     end
% end
% 
% Hmat(length(x)+length(x)*d+(1:d2),length(x)+length(x)*d+(1:d2)) = sum_Zsq_times_k;
% Hmat(1:length(x)                 ,length(x)+length(x)*d+(1:d2)) = sum_Z_times_k;
% Hmat(length(x)+length(x)*d+(1:d2),(1:length(x)))                = sum_Z_times_k';
% for k = 1:d,
%     Hmat(length(x)+(1:length(x)*k)   ,length(x)+length(x)*d+(1:d2)) = sum_diff_X_times_Z(:,k,:);
%     Hmat(length(x)+length(x)*d+(1:d2),length(x)+(1:length(x)*k)   ) = sum_diff_X_times_Z(:,k,:)';
% end
% 
% 
% fvec_alpha = sum_y_times_k;
% fvec_beta = reshape(sum_diff_X_times_yk,length(x)*d,1);
% fvec_gamma = sum_y_times_Zk;
% fvec = -[fvec_alpha;fvec_beta;fvec_gamma];

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
    Amat_afriat = sparse(length(row_index), length(x)*(d+1)+d2);

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
    Amat_afriat = sparse(n*(n-1), length(x)*(d+1));
    for ii = 1:n,
        for jj = 1:n,
            if ii~=jj,
                Amat_afriat(row,ii) = -1;
                Amat_afriat(row,jj) = 1;
                for k = 1:d
                    Amat_afriat(row,ii + n*k) = x(ii,k)-x(jj,k);
                end
                row = row + 1;

            end
        end
    end
end



Amat = Amat_afriat;
bvec=zeros(row-1,1);




end