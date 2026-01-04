function [WW,DD,S] = update_S(Hm,G,R, lambda1,mu,n)
	distx = [];
	distx = EuDist2(Hm',Hm',0);
	distx=distx-diag(diag(distx));
	beta=distx;
	beta=beta.*(lambda1*0.5);
	Z=G-R./mu;
	M=Z-beta./mu;
	S = zeros(n,n);
	temp_j=1:n;
	for j=1:n
		S(j,temp_j~=j) =opt_S(M(j,temp_j~=j));
	end
	
	WW = ((S')+S)/2; 
	DD = diag(sum(WW, 1)); 

end
