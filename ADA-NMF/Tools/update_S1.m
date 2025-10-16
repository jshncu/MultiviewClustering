function [WW,DD,S] = update_S1(Hm,G,R,lambda1,mu,n)
	distx = [];
	% distx = EuDist2(Hm',Hm',0);
	distx = L2_distance_1(Hm,Hm);
	distx=distx-diag(diag(distx));%这行代码的目的是将 distx 矩阵的对角线元素置为零
	beta=distx;
	beta=beta.*(lambda1*0.5);
	Z=G-R./mu;
	M=Z-beta./mu;
	S = zeros(n,n);
	temp_j=1:n;%创建一个从 1 到 n 的数组，并将其赋值给变量 temp_j。该数组代表从 1 到 n 的连续整数序列
	for j=1:n
		S(j,temp_j~=j) =opt_S(M(j,temp_j~=j)); %这段代码中的 temp_j~=j 表示一个逻辑条件，它会生成一个逻辑索引矩阵。这个矩阵的第 j 行会包含 temp_j 中除了第 j 个元素外的所有元素的逻辑值
	end
	
	
	WW = ((S')+S)/2; 
	DD = diag(sum(WW, 1)); 

end
