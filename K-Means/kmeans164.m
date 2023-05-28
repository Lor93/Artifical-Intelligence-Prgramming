function [idx, C,D] = kmeans164(X, k,nmaxiterations)
% X: data of size n x m
% n: number of example
% m: number of featres
% k: number of cluster
% niterationsmax: maxium number of iterations
%% Output
%idx: cluster index for each example
% C: matrix containgn the control for each cluster
% D: distance between an example to the different centroids

[n,m]=size(X);

idx = zeros(n,1);
C=zeros(k,m);
D=zeros(n,k);

tmp = randperm(n);
for i=1:k
   C(i,:)=X(tmp(i),:);
end

iteration =0;

idx_old=zeros(n,1);
finished=false;

while ((iteration<nmaxiterations) && ~finished)

    %Assign to each example the corresopoding cluster
    for i=1:n

        x=X(i,:);
        d=zeros(k,1); %contains the distance between example i 
        %and centroid of cluster j

        for j=1:k
            c=C(j,:);
            delta=x-c;
            d(j)=sqrt(delta*delta');
        end
        % Get the cluster with the minimum distance with x
        [vmin, argmin]= min(d);
        idx(i)=argmin;

    end

    %% Part 2
    % Adjsut the values of each centroids
    for j=1:k
        idx_cluster=find(idx == j);
        Xcluster=X(idx_cluster,:);
        C(j,:)=mean(Xcluster,1);
    end 
%Part 3: Check if we are done
    delta_idx=idx_old-idx;
    distance_idx=sqrt(delta_idx'*delta_idx);
    if distance_idx == 0
        finished = true;
    else
        idx_old=idx;
    end
    
    iterartion=iteration+1;

end    