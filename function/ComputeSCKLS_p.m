function [alpha1,beta1,yhat,Amat,row] = ComputeSCKLS_p(y,X,Hmat,fvec,Amat,bvec,x,v_pt,row,concavity,increase,p)

% Number of observations
n = size(y,1);

% Number of inputs
d = size(X,2); 


% Add violated constraints
Amat_new = zeros(size(v_pt,1),size(Amat,2));
for i = 1:size(v_pt,1)
    Amat_new(i,v_pt(i,1)) = -1;
    Amat_new(i,v_pt(i,2)) = 1;
    for k = 1:d
        Amat_new(i,v_pt(i,1) + length(x)*k) = x(v_pt(i,1),k)-x(v_pt(i,2),k);
    end
    row = row + 1;
end

bvec=zeros(row-1,1);
Amat = ([sparse(Amat);sparse(Amat_new)]);



% Set lower bound
lb = -inf(size(Amat,2),1);
ub = inf(size(Amat,2),1);

if increase == 1
    lb(length(x)+1:length(x)*(d+1)) = 0;
elseif increase == -1
    ub(length(x)+1:length(x)*(d+1)) = 0;
end

% Impose constraints on second partial derivative (Hessian matrix)
comb = nmultichoosek(1:d,2);
for ll = 1:size(comb,1)
    if all(comb(ll,:) == comb(ll,1))
        if concavity == 1
            ub(length(x)*(d+1)+length(x)*(ll-1)+1:length(x)*(d+1)+length(x)*(ll-1)+length(x))=0;
        elseif concavity == 0
            lb(length(x)*(d+1)+length(x)*(ll-1)+1:length(x)*(d+1)+length(x)*(ll-1)+length(x))=0;
        end
    end
end


%Solve using solver

%CPLEX

% [est] = cplexqp(Hmat,fvec,Amat,bvec,[],[],lb,[],[]);

% path through origin
% Aeq = [1,zeros(1,length(x)-1),zeros(1,length(x))];
% beq = 0;

%Quadprog

if concavity == 1,
    Amat_temp = Amat; 
elseif concavity == 0,
    Amat_temp = -Amat;
end

options = optimoptions('quadprog','Display','off');
warning('off')
% options = cplexoptimset('Display','off','Algorithm','auto');

[est] = quadprog(Hmat,fvec,Amat_temp,bvec,[],[],lb,ub,[],options);
%     [est] = cplexqp(Hmat,fvec,Amat_temp,bvec,[],[],lb,[],[]);

warning('on')

est = reshape(est,length(x),size(Amat,2)/length(x));
yhat = est(:,1);
beta1 = est(:,2:(1+d));
for i = 1:length(x),
    alpha1(i,:) = yhat(i) - beta1(i,:)*(x(i,:)).';
end

%Print out the number of terms in objective function and constraints we used in Afriat inequality
max_cnt_cnst = length(x)*length(x)-length(x); 
% fprintf('Number of constrains used: %d/%d\n',row-1,max_cnt_cnst);

end
