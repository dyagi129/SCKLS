function [yhat,alpha_hat,beta_hat] = ComputeLL_fixed(X,x,y,h,kernel)



%% Initial check
n = size(y,1);
% size returns a vector of length 2 with height and width of x
d = size(X,2); 
% Reassign d with the value of d(2) makes d a scalar with the value indicating the 
% number of input variables



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



%% Calculate kernel weights
prod_k_grd = ones(size(x,1),n);

for kk = 1:d,
    prod_k_grd = prod_k_grd .* kerf((repmat(X(:,kk)',size(x,1),1) - repmat(x(:,kk),1,n))./h(kk));
end


%% Local Linear Estimates on Evaluation points
yhat = zeros(size(x,1),1);
alpha_hat = zeros(size(x,1),1);
beta_hat = zeros(size(x,1),d);

for i = 1:size(x,1)        
    A = ([ones(n,1), X - repmat(x(i,:),n,1)])' * diag(prod_k_grd(i,:)) * ([ones(n,1), X - repmat(x(i,:),n,1)]);
    B = ([ones(n,1), X - repmat(x(i,:),n,1)])' * diag(prod_k_grd(i,:)) * y;
    
    while rcond(A) < 1e-12
        A = A + eye(1+d,1+d).*(1/n);
    end
    
    [L,U] = lu(A);
    Y = L\B;
    temp_est = U\Y;
    
%     temp_est = A\B;
    yhat(i,:) = temp_est(1);
    beta_hat(i,:) = temp_est(2:end);
    
    alpha_hat(i,:) = yhat(i,:) - beta_hat(i,:)*x(i,:)';
end    

% eps_hat = y - yhat;

end