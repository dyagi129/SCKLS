function [alpha1,beta1,delta1,yhat,Amat,row] = ComputeSCKLS_z(y,X,Z,Hmat,fvec,Amat,bvec,x,v_pt,row)

% Number of observations
n = size(y,1);

% Number of inputs
d = size(X,2); 
d2 = size(Z,2);

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
lb = [repmat(-inf,length(x),1);zeros(length(x)*d,1);repmat(-inf,d2,1)];


%Solve using solver

%CPLEX
% [est] = cplexqp(Hmat,fvec,Amat,bvec,[],[],lb,[],[]);

% path through origin
% Aeq = [1,zeros(1,length(x)-1),zeros(1,length(x))];
% beq = 0;

%Quadprog
options = optimoptions('quadprog','Display','off');
[est] = quadprog(Hmat,fvec,Amat,bvec,[],[],lb,[],[],options);

delta1 = est((length(x)*(d+1)+1):end);
est = reshape(est((1:length(x)*(d+1))),length(x),d+1);
yhat = est(:,1);
beta1 = est(:,2:(1+d));
for i = 1:length(x),
    alpha1(i,:) = yhat(i) - beta1(i,:)*(x(i,:)).';
end

%Print out the number of terms in objective function and constraints we used in Afriat inequality
max_cnt_cnst = length(x)*length(x)-length(x); 
fprintf('Number of constrains used: %d/%d\n',row-1,max_cnt_cnst);

end
