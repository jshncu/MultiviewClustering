function [C_Tensor] = TensorDANMF_normalize(original_X,B,A,W,U,H,layers,options)

    num_view=options.num_view;
	num_layer=options.num_layer;
    n=options.n;
	
    rho1=options.rho1;
	rho2=options.rho2;
    rho3=options.rho3;
    mu1=options.mu1;
	mu2=options.mu2;
	mu3=options.mu3;
    max_mu1=options.max_mu1;
	max_mu2=options.max_mu2;
	max_mu3=options.max_mu3;
	lambda1=options.lambda1;
	lambda2=options.lambda2;
	lambda3=options.lambda3;
    p=options.p;
   
	tolfun=options.tolfun;
	maxiter=options.maxiter;
	 
	
	
	W_Tensor=cell(1,num_layer);
	U_Tensor=cell(1,num_layer);
	H_Tensor=cell(1,num_layer);
	for layer=1:num_layer
        W_Tensor{1,layer}=cat(3,W{layer,:});
		U_Tensor{1,layer}=cat(3,U{layer,:});
		H_Tensor{1,layer}=cat(3,H{layer,:});
	end
	
  
    Hm_Tensor=H_Tensor{1,num_layer};

	[m,n]=size(B{1,1});
	
	%%Initialization
	E=cell(1,num_view);
	C=cell(1,num_view);
    Y1=cell(1,num_view);
	Y2=cell(1,num_view);
	
    for v=1:num_view
		E{1,v}=zeros(m,n);
		C{1,v}=B{1,v};
		Y1{1,v}=zeros(layers(num_layer),n);
		Y2{1,v}=zeros(m,n);
	end
	
	R = cell(1,num_view);%%R=Y3
	G = cell(1,num_view);
	
	for i = 1: num_view
		R{1,i} = zeros(m,n);
		G{1,i} = zeros(m,n);
	end

	J1_Tensor=Hm_Tensor;
	
	
	for v=1:num_view
		J1_Tensor(:,:,v)=zeros(layers(num_layer),n);
	end
	


    Y1_Tensor=cat(3,Y1{:,:});
	Y2_Tensor=cat(3,Y2{:,:});
	
	G_tensor=cat(3,G{:,:});
	R_tensor=cat(3,R{:,:});
	
	B_Tensor=cat(3,B{:,:});
	C_Tensor=cat(3,C{:,:});
	E_Tensor=cat(3,E{:,:});
	

	
	W_f=cell(1,num_layer);
	U_f=cell(1,num_layer);
	H_f=cell(1,num_layer);
	
	
	for layer=1:num_layer
	    W_f{1,layer}=fft(W_Tensor{1,layer},[],3);
		U_f{1,layer}=fft(U_Tensor{1,layer},[],3);
		%H_f{1,layer}=fft(H_Tensor{1,layer},[],3);
	end
	
	
	
	Isconverg = zeros(1,num_view);
	iter=1;

%%%%%%%%%%%%%%%%%%%%
% Fine-tuning
%%%%%%%%%%%%%%%%%%%%
while(sum(Isconverg) == 0)

	C_f=fft(C_Tensor,[],3);
	B_f=fft(B_Tensor,[],3);
	E_f=fft(E_Tensor,[],3);
    Hm_f=fft(Hm_Tensor,[],3);
	J1_f=fft(J1_Tensor,[],3);
	
	Y1_f=fft(Y1_Tensor,[],3);
	Y2_f=fft(Y2_Tensor,[],3);
	

	G_f=fft(G_tensor,[],3);
	R_f=fft(R_tensor,[],3);
	
	%%%%Update C
    for v =1:num_view
	    PP=U_f{1,1}(:,:,v);
		for i=2:num_layer
			PP=PP*U_f{1,i}(:,:,v);
		end
		PP=PP*Hm_f(:,:,v);
	    K1{v} = 2 * PP;
	    K2{v} = B_f(:,:,v)-E_f(:,:,v)+Y2_f(:,:,v)/mu2;
		K3{v} = G_f(:,:,v)-R_f(:,:,v)/mu3;
	end
	y = zeros(m, n);
    for v = 1:num_view
        for i = 1:n
		    y(:,i) = lambda1 * K1{v}(:,i) + mu2 * K2{v}(:,i) +mu3 * K3{v}(:,i);
            MM = (y(:,i))/ (2*lambda1 + mu2 + mu3);
            C{v}(:,i) = EProjSimplex_new(MM, 1);   
        end
        C_f(:,:,v)=C{v};
    end
	C_Tensor=ifft(C_f,[],3);
	
	
	
	%%%%%Update W U H 
	for v=1:num_view
		
		Q1=cell(1,num_layer);
		Q2=cell(1,num_layer);
		Q1{1,num_layer}=B_f(:,:,v);
		Q2{1,num_layer}=Hm_f(:,:,v);
		for i=num_layer-1:-1:1
			Q1{1,i}=W_f{1,i+1}(:,:,v)*Q1{1,i+1};
			Q2{1,i}=U_f{1,i+1}(:,:,v)*Q2{1,i+1};
		end

		for i=1:num_layer
		        Q1{1,i}=Q1{1,i}+1e-5 * eye(size(Q1{1,i}));
		        Q2{1,i}=Q2{1,i}+1e-5 * eye(size(Q2{1,i}));		        
			if i==1
				W_f{1,1}(:,:,v)=Hm_f(:,:,v)*pinv(Q1{1,1});
				U_f{1,1}(:,:,v)=C_f(:,:,v)*pinv(Q2{1,1});
			else
				W_f{1,i}(:,:,v)=pinv(P1)*Hm_f(:,:,v)*pinv(Q1{1,i});
				U_f{1,i}(:,:,v)=pinv(P2)*C_f(:,:,v)*pinv(Q2{1,i});
			end
			

			
			if i==1
				P1=W_f{1,1}(:,:,v);
				P2=U_f{1,1}(:,:,v);
			else
				P1=P1*W_f{1,i}(:,:,v);
				P2=P2*U_f{1,i}(:,:,v);
			end
			
		end
		
		Hm_f(:,:,v)=(2*lambda1*P2'*P2+(2+mu1)*eye(layers(num_layer)))\(2*P1*B_f(:,:,v)+2*lambda1*P2'*C_f(:,:,v)+mu1*(J1_f(:,:,v)-Y1_f(:,:,v)/mu1));
	end
	
	Hm_Tensor=ifft(Hm_f,[],3);
   
	
	
    %%%Update J1
	for v=1:num_view
		J1_Tensor(:,:,v)=max(Hm_Tensor(:,:,v)+Y1_Tensor(:,:,v)/mu1,0);
	end
    
	%%%Update G
    R_tensor = cat(3, R{:,:});%%%R=Y3
    temp_C = C_Tensor(:);
    temp_R = R_tensor(:);
	sX2 = [m, n, num_view];
	weight_vector = ones(1,num_view)';
	for v=1:num_view
		weight_vector(v)=lambda3;
	end
  
	[g, ~] = wshrinkObj_weight_lp(temp_C + 1/mu3*temp_R, weight_vector./mu3,sX2, 0,3,p);
    G_tensor = reshape(g, sX2);
	
	%%Update R
	temp_R = temp_R + mu3*(temp_C - g);
	R_tensor = reshape(temp_R , sX2);
	
	for v=1:num_view
		G{v} = G_tensor(:,:,v);
		R{v} = R_tensor(:,:,v);
	end
	
	
	%%%Update E
	F =[];
	for v =1:num_view
        F=[F;B_Tensor(:,:,v)-C_Tensor(:,:,v)+Y2_Tensor(:,:,v)/mu2];
    end
	[Econcat] = solve_l1l2(F,lambda2/mu2);
	
	ro_b =0;
    E_Tensor(:,:,1) =  Econcat(1:size(B{1},1),:);
    ro_end = size(B{1},1);
    for i=2:num_view
        ro_b = ro_b + size(B{i-1},1);
        ro_end = ro_end + size(B{i},1);
        E_Tensor(:,:,i) =  Econcat(ro_b+1:ro_end,:);
    end
	
	
    %%Update Y1,Y2
    temp_Hm=Hm_Tensor(:);
    temp_Y1=Y1_Tensor(:);
    j1=J1_Tensor(:);
    temp_Y1=temp_Y1+mu1*(temp_Hm-j1);


    temp_B=B_Tensor(:);
    temp_C=C_Tensor(:);
    temp_E=E_Tensor(:);
    temp_Y2=Y2_Tensor(:);
    temp_Y2=temp_Y2+mu2*(temp_B-temp_C-temp_E);


    sX1 = [layers(num_layer), n, num_view];

    Y1_Tensor=reshape(temp_Y1,sX1);
    Y2_Tensor=reshape(temp_Y2,sX2);
	
	
	
	mu1=min(rho1*mu1,max_mu1);
	mu2=min(rho2*mu2,max_mu2);
	mu3=min(rho3*mu3,max_mu3);
	
	Isconverg = ones(1,num_view);
	
	for v=1:num_view
		norm_V_J = norm(C_Tensor(:,:,v)-G_tensor(:,:,v),inf);
		if (abs(norm_V_J)>tolfun)
            Isconverg(v) = 0;
        end
	end
	
    fprintf('----processing iter %d, dnorm=%f--------\n', iter,norm_V_J);
	if(iter>maxiter)
		Isconverg  = ones(1,num_view);
    end
    iter=iter+1;
	
 
 
end
end

function error = cost_f(B_Tensor, C_Tensor,W_Tensor,U_Tensor, Hm_Tensor,num_view,num_layer)
   error = 0;
    for v=1:num_view
		
        error = error +norm(Hm_Tensor(:,:,v)-reconstruction(W_Tensor,v,num_layer)*B_Tensor(:,:,v), 'fro')+norm(C_Tensor(:,:,v)-reconstruction(U_Tensor,v,num_layer)*Hm_Tensor(:,:,v), 'fro');
    end

end

function [ out ] = reconstruction(U_Tensor,v,num_layer)
    out = U_Tensor{1,num_layer}(:,:,v);
    for k = num_layer-1:-1:1
        out =  U_Tensor{1,k}(:,:,v)*out;
    end
end




