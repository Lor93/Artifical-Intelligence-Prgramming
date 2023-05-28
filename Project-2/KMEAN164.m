function [idx, centers] = kmeans(X, k)

% Initialize the cluster centers randomly
[n, m] = size(X);
centers = X(randperm(n, k), :);

% Loop until convergence or maximum number of iterations is reached
    for iter = 1:100
        % Assign each example to the nearest cluster
        dist = pdist2(X, centers);
        [~, idx] = min(dist, [], 2);
    
        % Update the cluster centers
        for i = 1:k
            centers(i, :) = mean(X(idx == i, :));
        end
    end
end