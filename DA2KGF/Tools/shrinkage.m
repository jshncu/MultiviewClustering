    function z = shrinkage(y, m)
    % 对矩阵 y 的每一列进行操作
    for j = 1:size(y, 2)
        % 对第 j 列进行排序，并得到排序后的索引
        [~, idx] = sort(abs(y(:, j)), 'descend');
        % 将除了前 m 个绝对值最大的元素外的其他元素置为 0
        y(idx(m+1:end), j) = 0;
    end
    % 返回处理后的矩阵
    z = y;
end