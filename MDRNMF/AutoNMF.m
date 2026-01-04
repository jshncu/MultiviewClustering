function [H]=AutoNMF(X,W,V,U,H,layers,options)

	num_view=options.num_view;
	num_layer=options.num_layer;
	n=options.n;
	rho=options.rho;
	mu=options.mu;
	max_mu=options.max_mu;
	maxiter=options.maxiter;
	lambda1=options.lambda1;
	lambda2=options.lambda2;
	lambda3=options.lambda3;
	tolfun=options.tolfun;
	p=options.p;

	% miniter=101;
	
	S  = cell(1, num_view);
	WW = cell(1, num_view);
	DD = cell(1, num_view); 

	R = cell(1,num_view);
	G = cell(1,num_view);
	
	%%初始化S
	for i = 1: num_view
		R{1,i} = zeros(n,n);
		G{1,i} = zeros(n,n);
		[WW{i},DD{i},S{i}] = update_S(H{i,num_layer},G{i},R{i}, lambda2,mu,n);
	end
	

	
	
	Isconverg = zeros(1,num_view);
	iter=1;

while(sum(Isconverg) == 0)
	% fprintf('----processing iter %d--------\n', iter);
	for v=1:num_view
		Q = cell(1, num_layer + 1);
		
		Q{num_layer + 1} = eye(layers(num_layer));
		for i_layer = num_layer:-1:2
			Q{i_layer} = Q{i_layer + 1}*V{v,i_layer};
		end
		
		
		Q2 = cell(1, num_layer + 1);
		
		Q2{num_layer + 1} = eye(layers(num_layer+2));
		for i_layer = num_layer:-1:3
			Q2{i_layer} = U{v,i_layer}*Q2{i_layer + 1};
		end
		
		for i=1:num_layer
			
			%%更新V(1<=i<=num_layer)
			if i==1
				A=W{v,num_layer}*Q{2};
				upper=A'*X{v};
				down=A'*A*V{v,1};
				V{v,i}=V{v,i}.*upper./max(down,1e-10);
			else
				A=W{v,num_layer}*Q{i+1};
				upper=A'*X{v}*P';
				down=A'*A*V{v,i}*P*P';
				V{v,i}=V{v,i}.*upper./max(down,1e-10);
			end
			
			if i == 1
				P = V{v,i};
			else
				P = V{v,i}*P;
			end
			
			
			if i == 2
				P2 = U{v,i};
			end
			if i>2
				P2 = P2*U{v,i};
			end
			
			%%%更新W
			if i==num_layer
				D=P2*H{v,num_layer};
				upper=X{v}*P'+lambda1*X{v}*D';
				down=W{v,num_layer}*P*P'+lambda1*W{v,num_layer}*D*D';
				W{v,i}=W{v,i}.*upper./max(down,1e-10);
			end
			
		end
		
		
		
		
		for i=1:num_layer
			
			%%更新U(2<=i<=num_layer)
			if i==2
				F=Q2{3}*H{v,num_layer};
				E=W{v,num_layer};
				upper=E'*X{v}*F';
				down=E'*E*U{v,i}*F*F';
				U{v,i}=U{v,i}.*upper./max(down,1e-10);
			end
			
			if i>2
				F=Q2{i+1}*H{v,num_layer};
				E=W{v,num_layer}*P2;
				upper=E'*X{v}*F';
				down=E'*E*U{v,i}*F*F';
				U{v,i}=U{v,i}.*upper./max(down,1e-10);
			end
			
			
			
			if i == 1
				P = V{v,i};
			else
				P = V{v,i}*P;
			end
			
			
			if i == 2
				P2 = U{v,i};
			end
			if i>2
				P2 = P2*U{v,i};
			end
			
			%%%更新H
			if i==num_layer
				GG=W{v,num_layer}*P2;
				upper=GG'*X{v}+lambda2*H{v,i}*WW{v};
				down=GG'*GG*H{v,num_layer}+lambda2*H{v,i}*DD{v};
				H{v,i}=H{v,i}.*upper./max(down,1e-10);
			end
		
		end
		
		[WW{v},DD{v},S{v}] = update_S(H{v,num_layer},G{v},R{v}, lambda2,mu,n);
		
	end
	% obj=caculate_obj(X,W,P,P2,H,num_layer,num_view);
	% fprintf('----processing iter %d, dnorm=%f--------\n', iter,obj);
	
	%%更新G
	S_tensor = cat(3, S{:,:});
    	R_tensor = cat(3, R{:,:});
    	temp_S = S_tensor(:);
    	temp_R = R_tensor(:);
	sX = [n, n, num_view];
	
	weight_vector = ones(1,num_view)';
	for v=1:num_view
		weight_vector(v)=lambda3;
	end
    	%twist-version
    	%[g, objV] = Gshrink(temp_S + 1/mu*temp_R,(n*lambda2)/mu,sX,0,3);
	[g, ~] = wshrinkObj_weight_lp(temp_S + 1/mu*temp_R, weight_vector./mu,sX, 0,3,p);
    G_tensor = reshape(g, sX);
	
	%%更新R
	temp_R = temp_R + mu*(temp_S - g);
	R_tensor = reshape(temp_R , sX);
	
	
	%%更新mu
	mu = min(rho*mu, max_mu);
	
	for v=1:num_view
		G{v} = G_tensor(:,:,v);
		R{v} = R_tensor(:,:,v);
	end
	
	
	
	
	Isconverg = ones(1,num_view);
	
	for v=1:num_view
		norm_V_J = norm(G_tensor(:,:,v)-S_tensor(:,:,v),inf);
		if (abs(norm_V_J)>tolfun)
            Isconverg(v) = 0;
        end
	end
	% dnormarray(iter)=(norm_V_J);
	%fprintf('----processing iter %d, dnorm=%f--------\n', iter,norm_V_J);
	if(iter>maxiter)
		Isconverg  = ones(1,num_view);
	end
	
	% if(iter<miniter)
		% Isconverg  = zeros(1,num_view);
	% end
	if iter>40
		iter=maxiter;
	end
	
	iter=iter+1;
	
end



end


function [obj]=caculate_obj(X,W,P,P2,H,num_layer,num_view)
	obj=norm(X{num_view}-W{num_view,num_layer}*P,'fro')^2+norm(X{num_view}-W{num_view,num_layer}*P2*H{num_view,num_layer},'fro')^2;
end