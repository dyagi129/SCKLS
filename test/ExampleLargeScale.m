mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../function'));


%% DGP (Cobb-douglas)
n = 10000;
X=unifrnd(1,10,n,2);
fun_DGP = @(x) x(:,1).^(0.4).*x(:,2).^(0.4);

y_true = fun_DGP(X);
y=y_true+normrnd(0,5,n,1);

%% Bandwidth selection for fixed and variable bandwidth
h = BandwidthFixed(X,y,'gaussian','Rule-of-Thumb');


%% SCKLS estimates
[yhat,alpha_hat,beta_hat,x2] = SCKLS(X,y,'fixed','percentile','gaussian',100,h,[],1,1);
yhat_obs = functional_estimate(alpha_hat,beta_hat,X,0);



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
plot3(x2(:,1),x2(:,2),yhat,'.')
surf(XX,YY,ZZ,repmat(10,size(XX,1),size(XX,2)),'linestyle', 'none');
alpha(0.2)
legend('SCKLS','True Function')