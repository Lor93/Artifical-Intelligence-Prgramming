close all
clc 
clear 

load('mnist_test.mat');
X=test_data';
Y=test_label';

%MLP function
load('data_mnist_test_original.mat');
load('data_mnist_train_original.mat');

%% Take First 500 Examples

num_samples = 500;

train_imgs = imgs(:, :, 1:num_samples, :);
train_labels = labels(1:num_samples, :);

test_imgs = imgs(:, :, num_samples+1:2*num_samples, :);
test_labels = labels(num_samples+1:2*num_samples, :);

%% Normalize Data

train_imgs = train_imgs / 255;
test_imgs = test_imgs / 255;

Test_mlp=1;
Test_cnn=1;

% Reshape training and testing datasets
train_imgs_k = reshape(train_imgs, [], size(train_imgs, 3))';
test_imgs_k = reshape(test_imgs, [], size(test_imgs, 3))';


%Normalize the data
X = zscore(X);

%Define k values to test
kvalues = [10, 20, 30];
kvalue = [1,5,10];
verbose=1;


% % Both loop can run at the same time but it will take awhile 
% % Loop through each k value and assess performance
for k = kvalue
    [Yout, Yconfidence] = KNN164(X,Y,X,k);
    ConfusionMatrix = confusionmat(Y,Yout);
    acc=100*trace(ConfusionMatrix)/length(Y);
    fprintf('Accuracy for k=%d: %.2f%%\n', k, acc);
end

%Loop through each k value and assess performance for the K-means
for k = kvalues
    [Yout, Yconfidence] = KMEAN164(X,k);
    ConfusionMatrix = confusionmat(Y,Yout);
    acc=100*trace(ConfusionMatrix)/length(Y);
    fprintf('Accuracy for k=%d: %.2f%%\n', k, acc);
end


if Test_mlp==1

%% Multilayer Perceptron
fprintf('\n MLP: \n')

num_hidden_values = [50, 100];
nnetworks=size(num_hidden_values,2);
num_classes = 10;

% Train MLP
    for indnetwork=1:nnetworks
        hiddenlayersize=num_hidden_values(indnetwork);
        mlp = fitcnet(train_imgs_k,train_labels,'LayerSizes',hiddenlayersize);
        [predicted_labels,Score]=predict(mlp,test_imgs_k);
        accuracy = sum(predicted_labels == test_labels) / length(test_labels);
        fprintf(' %d neurons: %.2f%%\n\n', hiddenlayersize, accuracy*100);
    end


end

if Test_cnn==1
%% CNN

% Reshape Data so it is 28 x 28 x 1 x 500
train_imgs = reshape(train_imgs, [28, 28, 1, size(train_imgs, 3)]);
test_imgs = reshape(test_imgs, [28, 28, 1, size(test_imgs, 3)]);


train_imgs1(:,:,1,:)=train_imgs;
test_imgs1(:,:,1,:)=test_imgs;

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(5, 20)
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Train CNN
options = trainingOptions('sgdm', 'MaxEpochs', 10, 'InitialLearnRate', 0.01);
cnn = trainNetwork(train_imgs1, categorical(train_labels), layers, options);

% Test CNN
predicted_labels = classify(cnn, test_imgs1);
accuracy = sum(predicted_labels == categorical(test_labels)) / numel(test_labels);
fprintf('Accuracy for CNN: %.2f%%\n', accuracy * 100);


end

