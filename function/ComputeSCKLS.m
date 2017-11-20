function [alpha1,beta1,yhat,Amat,row] = ComputeSCKLS(y,X,Hmat,fvec,Amat,bvec,x,v_pt,row,concavity,increase)

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
lb = [repmat(-inf,length(x),1);zeros(length(x)*d,1);repmat(-inf,size(Amat,2)-length(x)*(d+1),1)];
ub = [repmat( inf,length(x),1);zeros(length(x)*d,1);repmat(inf,size(Amat,2)-length(x)*(d+1),1)];

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
if increase == 1,
    [est] = quadprog(Hmat,fvec,Amat_temp,bvec,[],[],lb,[],[],options);
%     [est] = cplexqp(Hmat,fvec,Amat_temp,bvec,[],[],lb,[],[]);
elseif increase == -1,
    [est] = quadprog(Hmat,fvec,Amat_temp,bvec,[],[],[],ub,[],options);
%     [est] = cplexqp(Hmat,fvec,Amat_temp,bvec,[],[],[],ub,[]);
else
    [est] = quadprog(Hmat,fvec,Amat_temp,bvec,[],[],[],[],[],options);
%     [est] = cplexqp(Hmat,fvec,Amat_temp,bvec,[],[],[],[],[]);
end
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
