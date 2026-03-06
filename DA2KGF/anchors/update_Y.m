function Y = update_Y(Y,G, num_sample, num_cluster)
yg = diag(Y'* G)';
yy = diag(Y'*Y+eps*eye(num_cluster))';
obj1 = norm(G - bsxfun(@rdivide, Y, sqrt(yy)), 'fro')^2;
for t = 1 : 100
	% tmp = zeros(size(Y));
	% for i = 1 : num_sample
		% gi = G(i,:);
		% yi = Y(i,:);
		% si = (yg+gi.*(1-yi))./sqrt(yy+1-yi) - (yg-gi.*yi)./sqrt(yy-yi);
		% tmp(i, :) = si;
		% [~,index] = max(si(:));
		% Y1(i,:) = 0;
		% Y1(i,index) = 1;
	% end

	tmp1 = bsxfun(@plus, yg, G .* (1 - Y));
	tmp2 = bsxfun(@plus, yy, 1 - Y);
	tmp3 = bsxfun(@minus, yg, G .* Y);
	tmp4 = bsxfun(@minus, yy, Y);
	S = tmp1 ./ sqrt(tmp2) - tmp3 ./ sqrt(tmp4);
	[val, y] = max(S, [], 2);
	Y2 = full(sparse(1 : num_sample, y, 1, num_sample, num_cluster));
	yg = diag(Y2'* G)';
	yy = diag(Y2'*Y2+eps*eye(num_cluster))';
	obj2 = norm(G - bsxfun(@rdivide, Y2, sqrt(yy)), 'fro')^2;
    % fprintf('t: %d, %f - %f = %f\n', t, obj1, obj2, obj1 - obj2);
	if (obj1 > obj2)        
        Y = Y2;
        obj1 = obj2;
    else
        break;
    end    
end
end



