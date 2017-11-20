%% Add path
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../function'));


%% Daga Generation Process (Cobb-douglas)
X=unifrnd(1,10,100,2);
y=X(:,1).^0.4.*X(:,2).^0.4+normrnd(0,0.7,100,1);


%% Bandwidth selection for fixed and variable bandwidth
h = BandwidthFixed(X,y,'gaussian','Leave-one-out CV');
K = BandwidthKNN(X,y,'gaussian','Leave-one-out CV');


%% SCKLS estimates

% Fixed bandwidth
[yhat,alpha_hat,beta_hat,x] = SCKLS(X,y,'fixed','percentile','gaussian',100,h,[],1,1);
% Variable bandwidth
[yhat2,alpha_hat2,beta_hat2,x2] = SCKLS(X,y,'variable','percentile','gaussian',100,K,[],1,1);

% [p_value] = shape_test(X,y,100,100,1,1);

% Compute functional estimate
yhat_obs = functional_estimate(alpha_hat,beta_hat,X,0);
yhat_obs2 = functional_estimate(alpha_hat2,beta_hat2,X,0);
%% Output graph


[XX,YY] = meshgrid(min(X(:,1)):0.1:max(X(:,1)),min(X(:,2)):0.1:max(X(:,2)));
ZZ = zeros(size(XX,1),size(XX,2));
for ii = 1:size(XX,1)
    for jj = 1:size(XX,2)
        ZZ(ii,jj) = fun_DGP([XX(ii,jj),YY(ii,jj)]);
    end
end

figure()
axis tight; hold on
plot3(x(:,1),x(:,2),yhat,'.')
plot3(x2(:,1),x2(:,2),yhat2,'.')
surf(XX,YY,ZZ,repmat(10,size(XX,1),size(XX,2)),'linestyle', 'none');

legend('SCKLS-fixed','SCKLS-variable','True function')