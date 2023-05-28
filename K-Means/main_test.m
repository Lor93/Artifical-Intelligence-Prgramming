close all
clc
clear


%img=imread('turtles.jpg');

%imshow(img);

%[nw,nx,nc]=size(img);


% Example with handwritten digits
load('mnist_test.mat');
%load('turtles.jpg');
X=test_data';
aaa=1;

%Call the Kmeans
niterationsmax=20;
k=10;
[idx,C,D] = kmeans164(X,k,niterationsmax);
figure(2);
%for each cluster, find the example assigned to cluster i (1<=i<=k)
for i=1:k
    
    idx_cluster=find(idx==i); % index of all the example in X
    %scatter(X(idx_cluster,1),X(idx_cluster,2),vcolor(i)); hold on
% display the centroid
    img = C(i,:);
    img=reshape(img,28,28);
    subplot(1,k,i);
    imagesc(img);

       
    %scatter(C(i,1),C(1,2),[ vcolor(i) '+'],'LineWidth', 4);
end

fprintf('done\n')

%Toy dataset with 4 classes

ngroups=4;
n=500; %number of example per group
m=2; % 2 features per example 2d points (x,y)

v1=1;
v2=6;

X=zeros(n*ngroups, m);

% Group 1
X(1:n,1)=v1+randn(n,1);
X(1:n,2)=v1+randn(n,1);
% Group 2
X(n+1:2*n,1)=v2+randn(n,1);
X(n+1:2*n,2)=v1+randn(n,1);
%Group 3
X(2*n+1:3*n,1)=v1+randn(n,1);
X(2*n+1:3*n,2)=v2+randn(n,1);
%Group 4
X(3*n+1:end,1)=v2+randn(n,1);
X(3*n+1:end,2)=v2+randn(n,1);

figure(1);
scatter(X(:,1),X(:,2));


%Call the Kmeans
niterationsmax=20;
k=4;
[idx,C,D] = kmeans164(X,k,niterationsmax);

vcolor=['r' 'b' 'g' 'c'];

%figure(2);
%for each cluster, find the example assigned to cluster i (1<=i<=k)
for i=1:k
    
    idx_cluster=find(idx==i);
    scatter(X(idx_cluster,1),X(idx_cluster,2),vcolor(i)); hold on

    scatter(C(i,1),C(1,2),[ vcolor(i) '+'],'LineWidth', 4);
end





