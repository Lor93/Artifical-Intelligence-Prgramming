close all
clear
clc

%load("fisheriris-1.mat")
load ("iris_dataset.mat");
nclasses=3;

% Change k to 5 or 10
k=5;

%Change option 1 or option2 to 1 k is 10 or 0 when k is 5
opt1=0;
opt2=0;
X=meas;
Y=[ ones(50,1); 2*ones(50,1); 3*ones(50,1) ];


% vk=[5,10];
% nk=size(vk,2);

[n,m]=size(meas);
idx_total = [1:n];
ConfusionMatrix=zeros(nclasses, nclasses);

for i=1:n
    Xtest=X(i,:);
    Ytest=Y(i);
    idx_train=setdiff(idx_total, i);
    Xtrain=X(idx_train,:);
    Ytrain=Y(idx_train);
    [Yout, Yconfidence] = KNNCSci164(Xtrain, Ytrain, Xtest,k,opt1,opt2); 
    ConfusionMatrix(Ytest, Yout)=ConfusionMatrix(Ytest,Yout) + 1;

end

acc=100*trace(ConfusionMatrix)/n;



% With the option of K = 5
% opt1 = 0 and opt2 = 1 Acc = 94.6667
% opt1 = 0 and opt2 = 0 Acc = 96.6667
% opt1 = 1 and opt2 = 1 Acc = 94.6667
% opt1 = 1 and opt2 = 0 Acc = 96.6667


% With the OPtion of k = 10 
% opt1 = 0 and opt2 = 0 Acc = 96.6667
% opt1 = 0 and opt2 = 1 Acc = 95.3333
% opt1 = 1 and opt2 = 0 Acc = 96
% opt1 = 1 and opt2 = 1 Acc = 96










































