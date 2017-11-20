function yhat = functional_estimate(alpha_hat,beta_hat,X,convexity)

n = size(X,1);
d = size(X,2);

if convexity == 0,
    fun_est = @(x) min(alpha_hat + sum(beta_hat.*repmat(x,length(beta_hat),1),2));
elseif convexity == 1,
    fun_est = @(x) max(alpha_hat + sum(beta_hat.*repmat(x,length(beta_hat),1),2));
end
    
yhat = zeros(n,1);
for i = 1:n
    yhat(i,1) = fun_est(X(i,:));
end

end