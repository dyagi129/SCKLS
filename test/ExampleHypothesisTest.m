%% Add path
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../function'));


%% Daga Generation Process 
% Case 1 - H_0 is true: logarithm with multiplicative errors
% Case 2 - H_1 is true: S-shape with multiplicative errors

X=unifrnd(1,10,100,1);
y_h0=log(X).*exp(normrnd(0,0.1,100,1));
y_h1=1./(1+exp(-5.*log(.2.*X))).*exp(normrnd(0,0.1,100,1));


%% SCKLS estimates
B = 200;
[p_value_h0] = shape_test(X,y_h0,B);
[p_value_h1] = shape_test(X,y_h1,B);

