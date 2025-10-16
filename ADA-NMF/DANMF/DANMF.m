function [S,H] = DANMF(XX,W,U,H,layers,options)

    numOfView = numel(XX);
    num_of_layers = numel(layers);
    n = size(XX{1,1},2);

    p=options.p;
    beta=options.beta;
    alpha=options.alpha;
    lambda=options.lambda;
    
    rho=options.rho;
    mu=options.mu;
    max_mu=options.max_mu;
    eta1=options.eta1;
    max_eta1=options.max_eta1;
   
    tolfun=options.tolfun;
    maxiter=options.maxiter;
  

    S  = cell(1, numOfView);
    WW = cell(1, numOfView);
    DD = cell(1, numOfView);
    R = cell(1,numOfView);
    G = cell(1,numOfView);
    Y1 = cell(1,numOfView);
    E  = cell(1, numOfView);
   
    
    Isconverg = zeros(1,numOfView);
    iter=1;

    X_hat = XX;
    for i = 1:numOfView
	    R{1,i} = zeros(n,n);
	    G{1,i} = zeros(n,n);
        Y1{1,i}= zeros(size(XX{i},1),n);
        E{1,i}= zeros(size(XX{i},1),n);
    end

k=10;

distX=cell(1,numOfView);
distX1=cell(1,numOfView);
idx=cell(1,numOfView);
for v=1:numOfView
	distX{v} = L2_distance_1(H{v,num_of_layers},H{v,num_of_layers});
    [distX1{v}, idx{v}] = sort(distX{v},2);
    S{v} = zeros(n);
    for i = 1:n
        di = distX1{v}(i,2:k+2);
        id = idx{v}(i,2:k+2);
        S{1,v}(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
	WW{v} = ((S{v}')+S{v})/2;
	DD{v} = diag(sum(WW{v}, 1));
end

%%%%%%%%%%%%%%%%%%%%
% Fine-tuning
%%%%%%%%%%%%%%%%%%%%

% \Psi -> P; \Phi -> Q
Q = cell(numOfView, num_of_layers + 1);
Q_hat = cell(numOfView, num_of_layers + 1);
while(sum(Isconverg) == 0)
   
    for v_ind = 1:numOfView
        X = XX{v_ind};
       % XXT = X * X';
        Q{v_ind,num_of_layers + 1} = eye(size(X,1));
        Q_hat{v_ind,num_of_layers + 1} = eye(layers(num_of_layers));
        for i = 1:num_of_layers
            if i == 1
                J_hat = U{v_ind,1};
            else              
                J_hat = J_hat * U{v_ind,i};
            end
           if i == num_of_layers
                %%update for X_hat
                X_hat{v_ind} = max( (2*J_hat * H{v_ind,num_of_layers} + eta1 * (X +  Y1{v_ind}/eta1 -E{v_ind}) )/(2+eta1),0);
            end  
        
        end
        
		D =  XX{v_ind} - X_hat{v_ind} + Y1{v_ind}/eta1;
        %%update for E
        [E{v_ind}] = prox_l1(D,lambda/eta1);
        %%update for Y1
        Y1{v_ind} = Y1{v_ind} + eta1 * (XX{v_ind} - X_hat{v_ind} - E{v_ind});

        for i_layer = num_of_layers:-1:2 
            Q_hat{v_ind,i_layer} = U{v_ind,i_layer} * Q_hat{v_ind,i_layer + 1};
            Q{v_ind,i_layer} = W{v_ind,i_layer} * Q{v_ind,i_layer + 1};
        end
        
        for i = 1:num_of_layers

            if i == 1
                QX = Q{v_ind,2} * X;
                down = W{v_ind,1} * QX * QX';
                upper = H{v_ind,num_of_layers} * QX';
                W{v_ind,i} = W{v_ind,i}.* upper ./ max(down, 1e-10);


                down_hat = U{v_ind,i} * (Q_hat{v_ind,2} * (H{v_ind,num_of_layers} * H{v_ind,num_of_layers}') *  Q_hat{v_ind,2}');%%up
                upper_hat =X_hat{v_ind}* (H{v_ind,num_of_layers}'* Q_hat{v_ind,2}');
                U{v_ind,i} = U{v_ind,i}.* upper_hat ./ max(down_hat, 1e-10);
            elseif i == num_of_layers
                 down = (P' * P) * (W{v_ind,i} * X) * X' ;
                upper = P' * H{v_ind,num_of_layers} * X' ;
                W{v_ind,i} = W{v_ind,i}.* upper ./ max(down, 1e-10);

            else
                down = ((P' * P) *( W{v_ind,i} * Q{v_ind,i + 1} * X ))*( X' * Q{v_ind,i + 1}');
                upper = P' * H{v_ind,num_of_layers} * X' * Q{v_ind,i + 1}';
                W{v_ind,i} = W{v_ind,i}.* upper ./ max(down, 1e-10);

                down_hat = P_hat' * P_hat * U{v_ind,i} * Q_hat{v_ind,i + 1} *(H{v_ind,num_of_layers} * H{v_ind,num_of_layers}') * Q_hat{v_ind,i + 1}';
                upper_hat = P_hat' * X_hat{v_ind} * H{v_ind,num_of_layers}' * Q_hat{v_ind,i + 1}';
                U{v_ind,i} = U{v_ind,i}.* upper_hat ./ max(down_hat, 1e-10);

            end
   
            if i == 1
                P =  W{v_ind,1};
                P_hat = U{v_ind,1};
            else
                P= P * W{v_ind,i};
                P_hat = P_hat * U{v_ind,i};
            end
            if i == num_of_layers
                %%update for H
                down_H = P_hat' * P_hat * H{v_ind,num_of_layers} + H{v_ind,num_of_layers} + alpha *  H{v_ind,num_of_layers} * DD{v_ind} ;
                upper_H = P_hat'* X_hat{v_ind} + P * X + alpha * H{v_ind,num_of_layers} * WW{v_ind} ;
                H{v_ind,i} = H{v_ind,i}.* upper_H ./ max(down_H,1e-10);
            end

        end
    end
 

for v_ind = 1:numOfView
        H_cell = H{v_ind,num_of_layers} ;
        G_cell = G{v_ind} ;
        R_cell = R{v_ind} ;

        %%update for S
        [WW_cell,DD_cell,S_cell]  = update_S1(H_cell,G_cell,R_cell,alpha,mu,n);
        
        WW{v_ind} = WW_cell;
        DD{v_ind} = DD_cell;
        S{v_ind} = S_cell;
end

    %%update for G
    S_tensor = cat(3, S{:,:});
    R_tensor = cat(3, R{:,:});
    temp_S = S_tensor(:);
    temp_R = R_tensor(:);
    sX = [n, n, numOfView];
    weight_vector = beta * ones(1, numOfView)';
	[g, ~] = wshrinkObj_weight_lp(temp_S + 1/mu*temp_R, weight_vector./mu,sX, 0,3,p);
    G_tensor = reshape(g, sX);
	
    %%update for R
	temp_R = temp_R + mu*(temp_S - g);
	R_tensor = reshape(temp_R , sX);

	
	%%update for mu
	mu = min(rho*mu, max_mu);
	%%update for eta1
    eta1 = min(rho*eta1, max_eta1);

	for v_ind=1:numOfView
		G{v_ind} = G_tensor(:,:,v_ind);
		R{v_ind} = R_tensor(:,:,v_ind);
    end
    Isconverg = ones(1,numOfView);
	
	for v_ind=1:numOfView
		norm_V_J1 = norm(G_tensor(:,:,v_ind)-S_tensor(:,:,v_ind),inf);
		if (abs(norm_V_J1)>tolfun)
            Isconverg(v_ind) = 0;
        end
    end

 	fprintf('----processing iter %d, dnorm1=%f--------\n', iter,norm_V_J1);

	if(iter>maxiter)
		Isconverg  = ones(1,numOfView);
	end
	
	iter=iter+1;
	
end

end

function error = cost_f(X,X_hat,W,U,H,LL,alpha)
out = H{numel(H)};
error = norm(X_hat - reconstructionDe(U,H), 'fro') + norm(out - reconstructionEn(W , X), 'fro') +alpha * trace(out*LL*out')  ;
end 

function [ out ] = reconstructionDe( Z, H )
    out = H{numel(H)};  %out初始化为Hp,即H中的最后一个元素
for k = numel(H) : -1 : 1
    out =  Z{k} * out; %out 代表经过多次矩阵乘法运算后的输出。
end
end

function [ out ] = reconstructionEn( W, X )  %[ out ] 表示 reconstructionEn 函数的返回值是一个矩阵，用 out 变量来存储结果。
out = X;
for k = numel(W) : -1 : 1
    out =  W{k} * out; %out 代表经过多次矩阵乘法运算后的输出。
end
end
