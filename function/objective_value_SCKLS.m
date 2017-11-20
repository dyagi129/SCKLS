%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
%
% This is the function to compute matrix for quadratic programming
% (H,f,A,b)
% min  1/2*x'*H*x + f'*x
% s.t. A*x <= b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [objective_value] = objective_value_SCKLS(X,y,h,kernel,grid_point,x,yhat_SCKLS,beta_SCKLS,p)

est = [yhat_SCKLS;beta_SCKLS(:)];

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



objective_value = est' * Hmat * est + 2 * fvec' * est + sum(prod_k * y.^2);

end