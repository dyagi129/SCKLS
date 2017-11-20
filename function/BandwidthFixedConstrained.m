%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function of Bandwidth Selection 
%
%
% Input arguments:
%       X:              observed input
%       y:              observed output
%       kernel:         type of kernel ('gaussian','uniform','epanechnikov')
%       selection:      method of bandwidth selection ('Rule-of-Thumb','Leave-one-out CV')       
%
% Output arguments:
%       h_opt:          optimal bandwidth
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [h_opt] = BandwidthFixedConstrained(X,y,kernel,selection)



%% Initial check
n = size(y,1);
d = size(X,2); 
l = zeros(n,d);



switch nargin,
    case [0,1],
        error('Not enough arguments.')
    case 2,
        kernel = 'gaussian';
        selection = 'Leave-one-out CV'; 
    case 3,
        selection = 'Leave-one-out CV';
end

if strcmp(kernel,'gaussian') + strcmp(kernel,'uniform') + strcmp(kernel,'epanechnikov') == 0,
    error('"kernel" has wrong name.')
end
if strcmp(selection,'Rule-of-Thumb') + strcmp(selection,'Leave-one-out CV') + strcmp(selection,'Leave-one-out CV LC') == 0,
    error('"type_bandwidth" has wrong name.')
end



%How many % of data can be used for the cross validation
if n < 300,
    PerCV = 1;
else
    PerCV = 100/n;
end



%% Define Kernel function
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



%% Function of Leave-one-out Cross Validation for Local Linear
function [MSE] = LOO_CV(h)
    MSE = 0;
    fun_est_LOO = zeros(n,1);
    
    
    prod_k = ones(length(index_sample),n);
   
    for kk = 1:d,
        prod_k = prod_k .* kerf((repmat(X(:,kk)',length(index_sample),1) - repmat(X(index_sample,kk),1,n))./h(kk));
    end

    for ii = 1:1:length(index_sample),
        
        [yhat,alpha_hat,beta_hat,x,perc_const] = SCKLS(X([1:(ii-1) ii+1:end],:),y([1:(ii-1) ii+1:end]),'fixed','equal','gaussian',100,h);
        yhat_obs = functional_estimate(alpha_hat,beta_hat,X(ii,:),0);
        
        MSE_temp = (y(index_sample(ii)) - yhat_obs).^2;
        
        if isnan(MSE_temp)
        else
            MSE = MSE + MSE_temp;
        end      
    end
end



%% Function of Leave-one-out Cross Validation for Local Constant
    function [MSE] = LOO_CV_LC(h)
        MSE = 0;
        
        prod_k = ones(n,n);
        for kk = 1:d,
            prod_k = prod_k .* kerf((repmat(X(:,kk)',n,1) - repmat(X(:,kk),1,n))./h(kk));
        end
        
        for i = 1:n        
            y_temp = y;
            y_temp(i,:) = [];
            prod_k_temp = prod_k(:,i);
            prod_k_temp(i) = [];
            temp_est = sum((y_temp .* repmat(prod_k_temp,1,size(y,2))))/max(eps,sum(prod_k_temp));
            ghat = temp_est;
            if isnan(ghat)
            else
                MSE = MSE + sum((y(i,:) - ghat).^2);
            end
        end    
    end



%% Find optimal bandwidth based on the option selected
if strcmp(selection,'Leave-one-out CV'),
    h0=std(X)*n^(-1/(4+d));
    index_sample = randsample(n,round(n*PerCV));
    options = optimset('Display','off');%,'MaxFunEvals',100);
    warning('off');
    [h_opt,fval]=fminsearch(@LOO_CV,h0,options);
    %h0
    h_opt_display = sprintf(' %.5g ',h_opt);
    fprintf('Optimal bandwidth: [%s]\n',h_opt_display);
elseif strcmp(selection,'Rule-of-Thumb'),
    h_opt = std(X)*n^(-1/(4+d));
elseif strcmp(selection,'Leave-one-out CV LC'),
    h0=std(X)*n^(-1/(4+d));
    options = optimset('Display','off');
    warning('off');
    [h_opt,fval]=fminsearch(@LOO_CV_LC,h0,options);
    %h0;
    %h_opt_display = sprintf(' %.5g ',h_opt);
%     fprintf('Optimal bandwidth: [%s]\n',h_opt_display);
end


end