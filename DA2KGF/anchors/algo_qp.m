function [Final_V1, Final_V2, obj, obj2] = algo_qp(X,m,k, A, Z, param)
% convergence conditions, maxiter = 200
num_view = length(X);
n = size(X{1,1},2);

alpha = param.alpha;
beta = param.beta;
gamma = param.gamma;
r = param.r;
k2 =  param.k2;
maxiter=param.maxiter;
tolfun=param.tolfun;

R = cell(1, num_view);
H = cell(1,num_view);
J = cell(1,num_view);
Pi = cell(1,num_view);
Theta = cell(1, num_view);
M  = cell(1, num_view);
C  = cell(1, num_view);

mu = 1e-7;
rho = 1.5;

%% initialize F A w

F = zeros(k2 * m, k2);

% assure F'F = I
for i = 1 : k2
    start_idx = (i - 1) * m + 1;
    end_idx = i * m;
    F(start_idx : end_idx, i) = 1;
    F = NormalizeFea(F,0);
end

% disp(F);
G = 0;
for iv = 1:num_view
	num_elements = k2 / k;
    for i = 1:k
        start_idx = (i - 1) * num_elements + 1;
        end_idx = i * num_elements;
        H{iv}(start_idx:end_idx, i) = 1;
    end
	H{iv} = NormalizeFea(H{iv},0);
    M{iv} = A{iv} * F;  
	G = G + H{iv}' * F' * Z{iv};
	J{iv} = 0;
	Pi{iv} = 0;
	R{iv} = 0;
	Theta{iv} = zeros(size(Z{iv}));
end

[Unew,~,Vnew] = svd(G,'econ');
G = Vnew * Unew'; 

for iv = 1 : num_view
	C{iv} = Z{iv} * G;
end

quadprog_options = optimset( 'Algorithm','interior-point-convex','Display','off');
iter = 1;
Isconverg = 0;

obj=[];
obj2 = [];
%% optimization
while (Isconverg == 0)
    %% update for A
    
    %%%
    % testobj1 = calobj(X, A, Z, F, H, G, M, w, num_view, alpha, beta,r);
	% fprintf('update for A\n');
    for iv = 1:num_view
        % obj1 = sum(sum((X{iv} - A{iv} * Z{iv}).^2)) + beta *sum(sum((A{iv}' - F * M{iv}').^2));
        A{iv} = (X{iv} * Z{iv}' + beta * M{iv} * F') / (Z{iv}*Z{iv}' + beta * eye(m*k2));
        % obj2 = sum(sum((X{iv} - A{iv} * Z{iv}).^2)) + beta *sum(sum((A{iv}' - F * M{iv}').^2));
        % fprintf('\t\t%f-%f=%f\n', obj1, obj2, obj1 - obj2);
    end
    %%%
    % testobj2 = calobj(X, A, Z, F, H, G, M, w, num_view, alpha, beta,r);
    %% update for Z
	% Z' 1 = 1
    % for iv = 1:num_view   
        % Z_tmp1 = 2 * A{iv}' * A{iv} + 2 * alpha * eye(m*k2);
        % Z_tmp1 = (Z_tmp1'+Z_tmp1)/2;
        
        % Z_tmp2 =  - 2 * X{iv}' * A{iv} - 2 * alpha * G * H{iv}' * F';
		% % tmpZ = Z{iv};
        % for ii=1:n
            % Z{iv}(:,ii) = quadprog(Z_tmp1, Z_tmp2(ii, :)', [], [], ones(1,m*k2), 1, zeros(m*k2,1), ones(m*k2,1), [], quadprog_options);%
        % end
		% %Z{iv} = tmpZ;
    % end
	% fprintf('update for Z\n');
	tmpG = 0;
	mu2 = mu / 2;
	for iv = 1 : num_view
		% update Z{iv}
		% obj1 = norm(X{iv} - A{iv} * Z{iv},'fro')^2 + alpha * norm( Z{iv} - F * H{iv} * G','fro')^2 + gamma * norm(Z{iv} - C{iv} * G', 'fro')^2 + mu2 * norm(R{iv} - Z{iv} + Theta{iv} / mu,'fro')^2;
		Z{iv} = (A{iv}' * A{iv} + (alpha + mu / 2 + gamma) * eye(m*k2)) \ (A{iv}' * X{iv} + (alpha * F * H{iv} + gamma * C{iv}) * G' + mu / 2 * R{iv} + Theta{iv} / 2);
		% obj2 = norm(X{iv} - A{iv} * Z{iv},'fro')^2 + alpha * norm( Z{iv} - F * H{iv} * G','fro')^2 + gamma * norm(Z{iv} - C{iv} * G', 'fro')^2 + mu2 * norm(R{iv} - Z{iv} + Theta{iv} / mu,'fro')^2;
		% fprintf('Z:\t\t%f-%f=%f\n', obj1, obj2, obj1 - obj2);
		
		
		% update R{iv}
		% obj1 = norm(R{iv} - Z{iv} + Theta{iv} / mu, 'fro')^2;
		tmpR = zeros(k2 * m, n);
		for ii = 1 : n
			tmpR(:, ii) = opt_S(Z{iv}(:, ii) - Theta{iv}(:, ii)/mu, 1);
		end
		R{iv} = tmpR;
		% obj2 = norm(R{iv} - Z{iv} + Theta{iv} / mu, 'fro')^2;
		% fprintf('S:\t\t%f-%f=%f\n', obj1, obj2, obj1 - obj2);

		%update H{iv}
		% obj1 = alpha * norm(Z{iv} - F * H{iv} * G','fro')^2 + mu2 * norm(J{iv} - H{iv} + Pi{iv} / mu,'fro');
		FZ = F' * Z{iv};
		tmp = 2 * alpha * FZ * G + mu * J{iv} + Pi{iv};
		err = any(any(isnan(tmp))) | any(any(isinf(tmp)));
		if err 
			return;
		end
		[Unew,~,Vnew] = svd(tmp, 'econ');
		H{iv} = Unew * Vnew';
		% obj2 = alpha * norm(Z{iv} - F * H{iv} * G','fro')^2 + mu2 * norm(J{iv} - H{iv} + Pi{iv} / mu,'fro');
		% fprintf('H:\t\t%f-%f=%f\n', obj1, obj2, obj1 - obj2);
		
		
		tmpG = tmpG + alpha * H{iv}' * FZ + gamma * C{iv}' * Z{iv};
		
		% update M{iv}
        M{iv} = A{iv} * F;   
		
		% update J{iv}
		J{iv} = max(H{iv} - Pi{iv}/mu, 0);
		
		% update Theta{iv} and Pi{iv}
		Theta{iv} = Theta{iv} + mu * (R{iv} - Z{iv});
		Pi{iv} = Pi{iv} + mu * (J{iv} - H{iv});
		
		
    end
	% obj1 = 0;
	% for iv = 1 : num_view
		% obj1 = obj1 + alpha * norm(Z{iv} - F * H{iv} * G','fro')^2 + gamma * norm(Z{iv} - C{iv} * G', 'fro')^2;
	% end
	[Unew,~,Vnew] = svd(tmpG,'econ');
	G = Vnew * Unew'; 
	% obj2 = 0;
	% for iv = 1 : num_view
		% obj2 = obj2 + alpha * norm(Z{iv} - F * H{iv} * G','fro')^2 + gamma * norm(Z{iv} - C{iv} * G', 'fro')^2;
	% end
	% fprintf('G:\t\t%f-%f=%.12f\n', obj1, obj2, obj1 - obj2);
	
	for iv = 1 : num_view
		C{iv} = Z{iv} * G;
	end
	% update mu
	mu = min(rho * mu, 1e10);
    %%%
    % testobj4 = calobj(X, A, Z, F, H, G, M, w, num_view, alpha, beta,r);
    %% update for G  
	% 1. G'G = I, G \ge 0
    % Gupper = 0;
    % Gdown = 0;
     % for iv = 1:num_view
        % wr = w(iv)^r;
        % Gupper = Gupper + wr * Z{iv}' * F * H{iv};
        % % Gdown = Gdown + wr * G * H{iv}' * F' * Z{iv} * G;
     % end
     % Gdown = G * Gupper' * G;
     % G = G .* (Gupper ./ max( Gdown , 1e-20));  
	% 2. G' G = I.
    % testobj5 = calobj(X, A, Z, F, H, G, M, w, num_view, alpha, beta,r);
	
    
    dnorm=0;
	obj2(iter) = 0;
	obj3(iter) = 0;
    for iv = 1:num_view
        dnorm = dnorm + norm(X{iv} - A{iv} * Z{iv},'fro')^2 + alpha * norm( Z{iv} - F * H{iv} * G','fro')^2 + beta * norm( A{iv}' - F * M{iv}','fro')^2 + gamma * norm(Z{iv} - C{iv} * G', 'fro')^2;
		obj2(iter) = obj2(iter) + norm(J{iv} - H{iv}, inf);
		obj3(iter) = obj3(iter) + norm(R{iv} - Z{iv}, inf);
    end
    obj(iter)=dnorm;
	
	% if iter >= 2
		% fprintf('obj(%d) - obj(%d) = %f\t\t obj2(%d) - obj2(%d) = %f\n', iter - 1, iter, obj(iter - 1) - obj(iter), iter - 1, iter, obj2(iter - 1) - obj2(iter));
	% end
	% if (abs((obj(iter-1) - obj(iter) ))<1e-4|| iter>maxiter)
        % Isconverg = 1;
    % end
	if ((iter >= 2 && abs((obj2(iter-1) - obj2(iter)))<1e-7 && abs((obj3(iter-1) - obj3(iter)))<1e-7 && abs((obj(iter-1)-obj(iter))/obj(iter - 1))<1e-4) || iter>maxiter)
        Isconverg = 1;
    end
	
    % if (iter>2) && (obj(iter-1) - obj(iter) < 0)
        % fprintf('\t\tupdate for A: %f - %f = %f\n', testobj1, testobj2, testobj1 - testobj2);
        % fprintf('\t\tupdate for Z: %f - %f = %f\n', testobj2, testobj3, testobj2 - testobj3);
        % fprintf('\t\tupdate for H and M: %f - %f = %f\n', testobj3, testobj4, testobj3 - testobj4);
        % fprintf('\t\tupdate for G: %f - %f = %f\n', testobj4, testobj5, testobj4 - testobj5);		
        % fprintf('\t\tupdate for w: %f - %f = %f\n', testobj5, testobj6, testobj5 - testobj6);
		% w'
        % pause();
    % end
	% fprintf('----processing iter %d, obj=%f--------\n', iter,dnorm);
	iter = iter + 1;
end
% 26
Vm_sum1=0;
for v=1:num_view
    Vm_sum1=Vm_sum1+ Z{v}';  %
end
Final_V1=Vm_sum1./num_view;
% 27
Final_V2=G;

end


% obj(iter) = calobj(X, A, Z, F, H, G, M, w, num_view, alpha, beta,r);
function obj = calobj(X, A, Z, F, H, G, M, w, num_view, alpha, beta,r)
obj = 0;
for iv = 1:num_view
    err1 = X{iv}-A{iv}*Z{iv};
    err2 =  Z{iv} - F* H{iv} * G';
    err3 = A{iv}' - F * M{iv}';
    obj = obj + w(iv)^r * (sum(sum(err1.*err1)) + alpha*sum(sum(err2.*err2))+beta*sum(sum(err3.*err3)));
end
end
