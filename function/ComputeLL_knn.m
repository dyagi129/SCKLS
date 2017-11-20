function [yhat,alpha_hat,beta_hat] = ComputeLL_knn(X,x,y,KNN,kernel,X_tilde,x_tilde)

yhat = [];
alpha_hat = [];
beta_hat = [];

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
Dist_grd = ones(size(x,1),n);

for kk = 1:d,
    Dist_grd = Dist_grd + (repmat(X_tilde(:,kk)',size(x,1),1) - repmat(x_tilde(:,kk),1,n)).^2;
end



%% Local Linear Estimates on Evaluation points
yhat = zeros(size(x,1),1);
R = zeros(size(x,1),1);

for i = 1:size(x,1),
   
    SortDist = sort(Dist_grd(i,:));
    SortDistRev = unique(nonzeros(SortDist));
    
    if KNN < 1,
        R(i) = SortDistRev(2);
    elseif KNN > length(SortDistRev),
        R(i) = SortDistRev(length(SortDistRev));
    else
        R(i) = SortDistRev(KNN);
    end
end
    
for i = 1:size(x,1),
    W = zeros(n,n);
    for j = 1:n,
       W(j,j) = kerf(Dist_grd(i,j)/R(i)); 
    end
        
    A = ([ones(n,1), X - repmat(x(i,:),n,1)])' * W * ([ones(n,1), X - repmat(x(i,:),n,1)]);
    B = ([ones(n,1), X - repmat(x(i,:),n,1)])' * W * y;
    
    while rcond(A) < 1e-12
        A = A + eye(1+d,1+d).*(1/n);
    end
    
    temp_est = A\B;
    yhat(i,:) = temp_est(1); 

end

% eps_hat = y-yhat;

end