function [Yout, Yconfidence] = KNN164(Xtrain, Ytrain, Xtest,k)

n = size(Xtrain, 1);
p = size(Xtest, 1);
Yout = zeros(p, 1);
Yconfidence = zeros(p, k);

% Calculate Euclidean distances between test set and training set
D = pdist2(Xtest, Xtrain);

    for i = 1:p
        % Find the k-nearest neighbors
        [sorted_dist, sorted_idx] = sort(D(i,:), 'ascend');
        k_nn_idx = sorted_idx(1:k);
        
        % Calculate the confidence of the k-nearest neighbors
        k_nn_labels = Ytrain(k_nn_idx);
        k_nn_dist = sorted_dist(1:k);
        Yconfidence(i,:) = 1./k_nn_dist;
        
        % Predict the class of the test example by taking the mode of the k-nearest neighbors
        Yout(i) = mode(k_nn_labels);
    end

end

