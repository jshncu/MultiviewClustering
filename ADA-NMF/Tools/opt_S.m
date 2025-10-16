function [x ft] = opt_S(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%

if nargin < 2 %用于返回当前函数被调用时传递给它的输入参数的个数。
    k = 1;
end

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;%%得到Mi
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;%%是0-1数组
        npos = sum(posidx);%%1的个数
        g = -npos;
        f = sum(v1(posidx)) - k;%%f是sTe-1(一个数)
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);%%v1中小于0的元素都换成0
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end
