%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to compute matrix for quadratic programming
% (H,f,A,b)
% min  1/2*x'*H*x + f'*x
% s.t. A*x <= b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Hmat,fvec,lb,ub] = ComputeMatrix_fixed_coordinate_LP(X,y,h,kernel,grid_point,x,dx)



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

Num_FP = d;
Num_SP = 1/2 * d * (d+1);

W = zeros(Num_FP,n,m);
WW = zeros(Num_SP,n,m);

for jj = 1:n
    for ii = 1:m
        for kk = 1:Num_FP
            W(kk,jj,ii) = (X(jj,kk)-x(ii,kk));
        end
        cnt = 1;
        for kk1 = 1:d
            for kk2 = kk1:d
                WW(cnt,jj,ii) = (X(jj,kk1)-x(ii,kk1)) * (X(jj,kk2)-x(ii,kk2));
                cnt = cnt + 1;
            end
        end
    end
end


sum_k = zeros(length(x),1);
sum_W_k = zeros(length(x),Num_FP);
sum_WW_k = zeros(length(x),Num_SP);
sum_W_W_k = zeros(length(x),Num_FP,Num_FP);
sum_W_WW_k = zeros(length(x),Num_FP,Num_SP);
sum_WW_WW_k = zeros(length(x),Num_SP,Num_SP);

sum_y_k = zeros(length(x),1);
sum_W_y_k = zeros(length(x),Num_FP);
sum_WW_y_k = zeros(length(x),Num_SP);

% Constant
sum_k = sum(prod_k,2);
sum_y_k = prod_k * y;
for i = 1:length(x),
    
    % First partial derivative
    for k = 1:Num_FP
        sum_W_k(i,k) = sum(W(k,:,i)' .* prod_k(i,:)');
        sum_W_y_k(i,k) = sum(W(k,:,i)' .*y .*prod_k(i,:)');
        for l = 1:Num_FP,
             sum_W_W_k(i,k,l) = sum(W(k,:,i)' .* W(l,:,i)' .* prod_k(i,:)');
        end
        for l = 1:Num_SP
            sum_W_WW_k(i,k,l) = sum(W(k,:,i)' .* WW(l,:,i)' .* prod_k(i,:)');
        end
    end
    
    % Second partial derivative
    for k = 1:Num_SP
        sum_WW_k(i,k) = sum(WW(k,:,i)' .* prod_k(i,:)');
        sum_WW_y_k(i,k) = sum(WW(k,:,i)' .*y .*prod_k(i,:)');
        for l = 1:Num_SP
            sum_WW_WW_k(i,k,l) = sum(WW(k,:,i)' .* WW(l,:,i)' .* prod_k(i,:)');
        end
    end
end


Hmat = zeros(length(x)+length(x)*Num_FP+length(x)*Num_SP);
for i = 1:length(x)
    % Constant
    Hmat(i,i) = sum_k(i);
    
    % First partial derivative
    for k = 1:Num_FP
        Hmat(i+k*m,i) = sum_W_k(i,k);
        Hmat(i,i+k*m) = sum_W_k(i,k);
        for l = 1:Num_FP
            Hmat(i+k*m,i+l*m) = sum_W_W_k(i,k,l);
        end
    end
    
    % Second partial derivative
    for k = 1:Num_SP
        Hmat(i+Num_FP*m+k*m,i) = sum_WW_k(i,k);
        Hmat(i,i+Num_FP*m+k*m) = sum_WW_k(i,k);       
        for l = 1:Num_FP
            Hmat(i+Num_FP*m+k*m,i+l*m) = sum_W_WW_k(i,l,k);
            Hmat(i+l*m,i+Num_FP*m+k*m) = sum_W_WW_k(i,l,k);
        end
        for l = 1:Num_SP
            Hmat(i+Num_FP*m+k*m,i+Num_FP*m+l*m) = sum_WW_WW_k(i,k,l);
        end
    end
end

fvec_cns = sum_y_k;
fvec_FP = reshape(sum_W_y_k,m*Num_FP,1);
fvec_SP = reshape(sum_WW_y_k, m*Num_SP,1);
fvec = -[fvec_cns;fvec_FP;fvec_SP];

% Amat_afriat = [];


%% Coordinate-wise concavity

ub_SP = zeros(Num_SP*m,1);
cnt = 1;

for k1 = 1:d
    for k2 = k1:d
        for i = 1:m
            if k1 == k2
                ub_SP(cnt) = 0;
            else
                ub_SP(cnt) = inf;
            end
            cnt = cnt + 1;
        end
    end
end


lb = [-inf(m,1); zeros(m*Num_FP,1); -inf(m*Num_SP,1)];
ub = [inf(m,1); inf(m*Num_FP,1); ub_SP];
% ub = [inf(m,1); inf(m*Num_FP,1); zeros(m*Num_SP,1)];

lb = [-inf(m,1); -inf(m*Num_FP,1); -ub_SP];
ub = [inf(m,1); zeros(m*Num_FP,1); inf(m*Num_SP,1)];
end