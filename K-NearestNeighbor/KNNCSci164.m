function [Yout,Yconfidence]=KNNCSci164(Xtrain,Ytrain,Xtest,k,opt1,opt2)

    [ntrain,nfeatures] = size(Xtrain);
    [ntest]=size(Xtest,1);
    Yout=zeros(ntest,1);
    Yconfidence=zeros(ntest,1);



    classes=unique(Ytrain);
    nclasses=size(classes,1);


    if opt2==1 
        Xtrain_mean=mean(Xtrain);
        Xtrain_std=std(Xtrain);

        Xtrain_mean1=repmat(Xtrain_mean,ntrain,1);
        Xtrain_std1=repmat(Xtrain_std, ntrain, 1);
        Xtest_mean1=repmat(Xtrain_mean,ntest,1);
        Xtest_std1=repmat(Xtrain_std, ntest, 1);

        Xtrain=(Xtrain-Xtrain_mean1)./Xtrain_std1;
        Xtest=(Xtest-Xtest_mean1)./Xtest_std1;
    end

    for itest=1:ntest
        x=Xtest(itest,:);
        Xtmp=repmat(x, ntrain, 1);

        %Euclidean Distance Calculation
        delta=Xtmp-Xtrain;
        delta=delta .* delta;
        d=sum(delta,2);
        [dsort,ix]=sort(d);
        v=zeros(nclasses,1);
   
        for i=1:k

            index_neighbor=ix(i); % index of the ith closest neighbor
            label = Ytrain(index_neighbor);

            if opt1 == 0
                v(label)=v(label)+1;
            elseif opt1 == 1
                v(label) = v(label) + (1 /( dsort(i) + 10e-5));
            end

        end

       
    vtotal=sum(v);
    v=v/vtotal;
    [Yconfidence(itest), Yout(itest)] = max(v);

    end
end


