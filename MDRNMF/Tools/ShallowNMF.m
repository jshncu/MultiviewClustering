function [U, V, dnorm] = ShallowNMF7777(X, r, maxiter, tolfun)

[m, n] = size(X);


V=eye(r,n);
U=X*V';

dnorm0 = norm(X - U * V, 'fro');

for i = 1:maxiter
    % update U
    
    U = U .* (X * V') ./ max(U * V * V', 1e-10);
    
    % update V
    
    V = V .* ( U' * X) ./ max(U' * U * V , 1e-10);
    
    dnorm = norm(X - U * V, 'fro');
    
    if abs(dnorm0 - dnorm) <= tolfun
        break; % converge
    end
    
    dnorm0 = dnorm;
    
end



dnorm = norm(X - U * V, 'fro');
