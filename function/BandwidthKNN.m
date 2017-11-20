%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the function of K-value selection for KNN
% written by Daisuke Yagi (d.yagi@tamu.edu)
%
%
% Input arguments:
%       X:              observed input
%       y:              observed output
%       kernel:         type of kernel ('gaussian','uniform','epanechnikov')
%       selection:      method of bandwidth selection ('Rule-of-Thumb','Leave-one-out CV')
%
% Output arguments:
%       K_opt:          optimal K-value
%
%
% For more information, please read the paper (http://~).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [K_opt] = BandwidthKNN(X,y,kernel,selection)



%% Initial check
n = size(y,1);
unique_n = length(unique(X,'rows'));
d = size(X,2); 
l = zeros(n,d);


%Error check
switch nargin,
    case [0,1],
        error('Not enough arguments.')
    case 2,
        kernel = 'gaussian';
        selection = 'Leave-one-out CV';
    case 3,
        selection = 'Leave-one-out CV';
    case 4,
end

if strcmp(kernel,'gaussian') + strcmp(kernel,'uniform') + strcmp(kernel,'epanechnikov') == 0,
    error('"kernel" has wrong name.')
end
if strcmp(selection,'Rule-of-Thumb') + strcmp(selection,'Leave-one-out CV') + strcmp(selection,'Generalized CV') + strcmp(selection,'Leave-one-out CV LC') == 0,
    error('"type_bandwidth" has wrong name.')
end


%How many % of data can be used for the cross validation
if n < 200,
    PerCV = 1;
else
    PerCV = 200/n;
end

PerCV=1;


%% Standarize data
X_tilde=(X-repmat(mean(X),n,1))./repmat(std(X),n,1);

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



%% Calculate distance for among observed points for CV
if strcmp(selection,'Leave-one-out CV'),
    index_sample = randsample(n,round(n*PerCV));
    Dist = zeros(length(index_sample),n);
    for kk = 1:d,
        Dist = Dist + (repmat(X_tilde(:,kk)',length(index_sample),1) - repmat(X_tilde(index_sample,kk),1,n)).^2;
    end
    Dist = sqrt(Dist);

elseif strcmp(selection,'Generalized CV') || strcmp(selection,'Leave-one-out CV LC'),
    Dist = zeros(n,n);
    for kk = 1:d,
        Dist = Dist + (repmat(X_tilde(:,kk)',n,1) - repmat(X_tilde(:,kk),1,n)).^2;
    end
    Dist = sqrt(Dist);
end



%% Function of Leave-one-out Cross Validation for Local Linear
    function [MSE] = LOOCV(k)  
        W = zeros(n-1,n-1);
        
  
        %Get k-nearest bandwidth
        for ii = 1:length(index_sample)
            SortDist = sort(Dist(ii,:));
            SortDistRev = unique(nonzeros(SortDist));
            if k < 1,
                R(ii) = SortDistRev(2);
            elseif k > length(SortDistRev),
                R(ii) = SortDistRev(length(SortDistRev));
            else
                R(ii) = SortDistRev(k);
            end
        end

        for ii = 1:length(index_sample),
            Xmat = [ones(n,1),X-repmat(X(index_sample(ii),:),n,1)];
            Xmat(index_sample(ii),:)=[];
            Yvec = y;
            Yvec(index_sample(ii))=[];
            ind = 1;
            for jj = 1:n,
                if jj~=index_sample(ii),
                    W(ind,ind) = kerf(Dist(ii,jj)/R(ii));
                    ind=ind+1;
                end
            end
            
            A = Xmat'*W*Xmat;
            B = Xmat'*W*Yvec;
            
%             while rcond(A) < 1e-12                
%                 A = A + eye(1+d,1+d).*(1/n);
%             end
            
            ghat(ii,:) = [1,zeros(1,d)]*(A\B);
        end  

        MSE=0;
        cnt = 0;
        for ii = 1:length(index_sample),
            if isnan(ghat(ii))
            else
                MSE = MSE + (y(index_sample(ii)) - ghat(ii))^2;
                cnt=cnt+1;
            end
        end
        MSE = MSE/cnt;
    end



%% Function of Leave-one-out Cross Validation for Local Constant
    function [MSE] = LOOCV_LC(k)
        
        W = zeros(n-1,1);
        
        
        %Get k-nearest bandwidth
        R = zeros(n,1);
        for ii = 1:n,
            SortDist = sort(Dist(ii,:));
            SortDistRev = unique(nonzeros(SortDist));
            if k < 1,
                R(ii) = SortDistRev(2);
            elseif k > length(SortDistRev),
                R(ii) = SortDistRev(length(SortDistRev));
            else
                R(ii) = SortDistRev(k);
            end
        end

        ghat = zeros(n,1);
        for ii = 1:n,
            Yvec = y;
            Yvec(ii,:)=[];
            Dist_temp = Dist(:,ii);
            Dist_temp(ii) = [];
            
            W = zeros(n-1,1);
            W(:,1) = kerf(Dist_temp/R(ii));
            ghat(ii,:) = sum((Yvec .* repmat(W,1,size(y,2))))/max(eps,sum(W));
        end  

        MSE=0;
        cnt = 0;
        for ii = 1:n,
            if isnan(ghat(ii))
            else
                MSE = MSE + (y(ii) - ghat(ii))^2;
                cnt=cnt+1;
            end
        end
        MSE = MSE/cnt;
    end



%% Find optimal K for K-nearest neighborhood
    
if strcmp(selection,'Rule-of-Thumb'),
    %% 10% of data
    K_opt = round(length(unique(X))*0.1);

    
    
    %% LOOCV for Local Constant estimator
elseif strcmp(selection,'Leave-one-out CV LC'),
        %% Algorithm to find optimal K by LOOCV
    obj_list = [];
    k_list = [];
    
    % When n is small, do exhaustive search
    if unique_n <= 200,
        for k = 1:200,
            [obj_list(k)]=LOOCV_LC(k); 
        end
        [curr_obj,K_opt] = min(obj_list);
    % When n is large, do exhaustive serach for small k (1-10),
    % and do raugh grid search for large k (11- ).
    else
        exp_v = 3/2; %higher 'exp' leads to more intensive search for small k
                   %1.1 - 2.0 is usual range to select
        for k = 1:10,
            [obj_list(k)] = LOOCV_LC(k);
            k_list(k) = k;
        end
        N_list = unique(round([1:1:100].^(exp_v) * (unique_n/(100^exp_v))));
        N_list(N_list<=10)=[];
        for iter = 1:length(N_list),
            k = k+1;
            [obj_list(k)] = LOOCV_LC(N_list(iter));
            k_list(k) = N_list(iter);
        end
        [curr_obj,K_opt] = min(obj_list);
    end 
    
    
    
elseif strcmp(selection,'Leave-one-out CV'),
    
    %% Algorithm to find optimal K by LOOCV
    obj_list = [];
    k_list = [];
    
    % When n is small, do exhaustive search
    if unique_n <= 200,
        for k = 1:200,
            [obj_list(k)]=LOOCV(k); 
        end
        [curr_obj,K_opt] = min(obj_list);
    % When n is large, do exhaustive serach for small k (1-10),
    % and do raugh grid search for large k (11- ).
    else
        exp_v = 3/2; %higher 'exp' leads to more intensive search for small k
                   %1.1 - 2.0 is usual range to select
        for k = 1:10,
            [obj_list(k)] = LOOCV(k);
            k_list(k) = k;
        end
        N_list = unique(round([1:1:100].^(exp_v) * (unique_n/(100^exp_v))));
        N_list(N_list<=10)=[];
        for iter = 1:length(N_list),
            k = k+1;
            [obj_list(k)] = LOOCV(N_list(iter));
            k_list(k) = N_list(iter);
        end
        [curr_obj,K_opt] = min(obj_list);
    end
    %At least using 10% of the data for very noisy data
    %K_opt = max(round(0.10*unique_n), K_opt);
    %Prevent over fitting
    %K_opt = min(round(0.50*unique_n), K_opt);
     
    fprintf('Optimal K is %d out of %d.\n',K_opt,unique_n);
end




end