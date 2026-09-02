function [model, results] = train_classifier(X, Y, patientID)

classNames = ["N" "A"];
Y = categorical(string(Y),classNames);          % Convert N/A labels to classes
patients = unique(patientID);                   % List of unique patients

numFolds = 5;
rng(1);                                         % Same CV split each run
cv = cvpartition(numel(patients),'KFold',5);

Sensitivity = zeros(numFolds,1);
PPV = zeros(numFolds,1);
F1 = zeros(numFolds,1);
Accuracy = zeros(numFolds,1);

miniBatchSize = 4096;

for fold = 1:numFolds

    trainPatients = patients(training(cv,fold));
    valPatients = patients(test(cv,fold));

    trainRows = ismember(patientID,trainPatients);
    valRows = ismember(patientID,valPatients);

    XTrain = X(trainRows,:);
    YTrain = Y(trainRows);
    XVal = X(valRows,:);
    YVal = Y(valRows);

    % Fill invalid values using TRAINING medians only
    imputeMedian = median(XTrain,1,'omitnan');
    imputeMedian(~isfinite(imputeMedian)) = 0;

    for feature = 1:size(XTrain,2)
        bad = ~isfinite(XTrain(:,feature));
        XTrain(bad,feature) = imputeMedian(feature);

        bad = ~isfinite(XVal(:,feature));
        XVal(bad,feature) = imputeMedian(feature);
    end

    % Standardise using TRAINING statistics only
    featureMean = mean(XTrain,1);
    featureStd = std(XTrain,0,1);
    featureStd(featureStd == 0) = 1;
    XTrain = (XTrain - featureMean) ./ featureStd;
    XVal = (XVal - featureMean) ./ featureStd;

    numFeatures = size(XTrain,2);
    numClasses = numel(classNames);
    layers = [
        featureInputLayer(numFeatures,Normalization="none")
        fullyConnectedLayer(128)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(64)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
    ];

    classCounts = countcats(YTrain);
    classWeights = numel(YTrain) ./ (numClasses * classCounts);
    lossFcn = @(scores,targets) crossentropy(scores,targets, ...
        Weights=classWeights);

    % Validation is reported once per epoch, but is not used for stopping
    % or selecting the network.
    iterationsPerEpoch = ceil(size(XTrain,1) / miniBatchSize);
    options = trainingOptions("adam", ...
        MaxEpochs=10, ...
        MiniBatchSize=miniBatchSize, ...
        InitialLearnRate=1e-3, ...
        Shuffle="every-epoch", ...
        ValidationData={XVal,YVal}, ...
        ValidationFrequency=iterationsPerEpoch, ...
        ValidationPatience=Inf, ...
        ExecutionEnvironment="auto", ...
        Verbose=true, ...
        VerboseFrequency=iterationsPerEpoch, ...
        Plots="none");

    % Train a temporary neural network for this fold
    net = trainnet(XTrain,YTrain,layers,lossFcn,options);

    % Predict patients not seen during training
    scores = minibatchpredict(net,XVal,MiniBatchSize=miniBatchSize);
    predicted = scores2label(scores,classNames);
    predicted = string(predicted);
    actual = string(YVal);

    TP = sum(predicted == "A" & actual == "A");
    FP = sum(predicted == "A" & actual == "N");
    FN = sum(predicted == "N" & actual == "A");
    TN = sum(predicted == "N" & actual == "N");

    if TP + FN > 0
        Sensitivity(fold) = TP / (TP + FN);
    end

    if TP + FP > 0
        PPV(fold) = TP / (TP + FP);
    end

    if Sensitivity(fold) + PPV(fold) > 0
        F1(fold) = 2 * Sensitivity(fold) * PPV(fold) / ...
            (Sensitivity(fold) + PPV(fold));
    end

    if TP + TN + FP + FN > 0
        Accuracy(fold) = (TP + TN) / (TP + TN + FP + FN);
    end

    fprintf("Fold %d: Sens = %.4f, PPV = %.4f, F1 = %.4f, Accuracy = %.4f\n", ...
        fold,Sensitivity(fold),PPV(fold),F1(fold),Accuracy(fold));

end

results = table((1:numFolds)',Sensitivity,PPV,F1,Accuracy, ...
    'VariableNames',{'Fold','Sensitivity','PPV','F1','Accuracy'});

fprintf("\n5-Fold Patient-Wise Cross Validation\n");
fprintf("Mean Sensitivity = %.4f\n",mean(Sensitivity));
fprintf("Mean PPV         = %.4f\n",mean(PPV));
fprintf("Mean F1          = %.4f\n",mean(F1));
fprintf("Mean Accuracy    = %.4f\n\n",mean(Accuracy));

% Train final model using every patient
imputeMedian = median(X,1,'omitnan');
imputeMedian(~isfinite(imputeMedian)) = 0;

for feature = 1:size(X,2)
    bad = ~isfinite(X(:,feature));
    X(bad,feature) = imputeMedian(feature);
end

featureMean = mean(X,1);
featureStd = std(X,0,1);
featureStd(featureStd == 0) = 1;
X = (X - featureMean) ./ featureStd;

numFeatures = size(X,2);
numClasses = numel(classNames);
layers = [
    featureInputLayer(numFeatures,Normalization="none")
    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
];

classCounts = countcats(Y);
classWeights = numel(Y) ./ (numClasses * classCounts);
lossFcn = @(scores,targets) crossentropy(scores,targets, ...
    Weights=classWeights);

iterationsPerEpoch = ceil(size(X,1) / miniBatchSize);
options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="auto", ...
    Verbose=true, ...
    VerboseFrequency=iterationsPerEpoch, ...
    Plots="none");

[classifier,trainingInfo] = trainnet(X,Y,layers,lossFcn,options);

model.classifier = classifier;
model.imputeMedian = imputeMedian;
model.featureMean = featureMean;
model.featureStd = featureStd;
model.classNames = classNames;
model.trainingInfo = trainingInfo;

if ~exist("Results","dir")
    mkdir("Results");
end
save(fullfile("Results","trained_model.mat"),"model","results","-v7.3");

fprintf("Final deep learning model saved to Results/trained_model.mat\n");

end
