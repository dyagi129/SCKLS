%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code was written by Daisuke Yagi (d.yagi@tamu.edu).
% This is the function of Partially Linear Model to calculate delta which 
% is the slope of the contextual variables Z
%
%
% Input arguments:
%       X:              observed input
%       y:              observed output
%       Z:              contexual variables
%       type_bandwidth: type of bandwidth ('fixed','variable')
%                       where I use KNN for variable bandwidth
%       kernel:         type of kernel ('gaussian','uniform','epanechnikov')
%
%
% Output arguments:
%       delta:          slope estimate of contextual variables
%       y_rev:          revised y values by getting rid of the effect of Z
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [delta,y_rev] = PartiallyLinear(X,y,Z,type_bandwidth,kernel)
    
    n = size(Z,1);
    d = size(Z,2);

    % Estimate conditional expectation by Local Constant Kernel
    y_given_X = LC(X,y,type_bandwidth,kernel);
    Z_given_X = zeros(n,d);
    for dd = 1:size(Z,2),
        Z_given_X(:,dd) = LC(X,Z(:,dd),type_bandwidth,kernel);
    end

    y_tilde = y - y_given_X;   
    Z_tilde = Z - Z_given_X;
    
    % Check whether Z'Z matrix is invertible
    ZZ = Z_tilde'*Z_tilde;
    while rcond(ZZ) < 1e-12
        ZZ = ZZ + eye(d,d).*(1/n);
    end
    
    % Calculate slope estimates by OLS
    delta = ZZ\(Z_tilde'*y_tilde);
    y_rev = y - Z * delta;
end

