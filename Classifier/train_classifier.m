function [model, results] = train_classifier(X, Y, patientID)

classNames = ["N" "A"];
Y = categorical(string(Y),classNames);
Y = Y(:);
patientID = patientID(:);
if any(isundefined(Y))
    error("Y must contain only the class labels N and A.");
end

patients = unique(patientID);
rng(1);
numFolds = 5;
cv = cvpartition(numel(patients),'KFold',numFolds);

numClasses = 2;
classWeightPower = 0.75;
miniBatchSize = 4096;
options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=3, ...
    LearnRateDropFactor=0.3, ...
    Shuffle="every-epoch", ...
    L2Regularization=1e-4, ...
    ExecutionEnvironment="auto", ...
    Verbose=true, ...
    Plots="none");

Sensitivity = zeros(numFolds,1);
PPV = zeros(numFolds,1);
F1 = zeros(numFolds,1);
Accuracy = zeros(numFolds,1);

for fold = 1:numFolds
    trainPatients = patients(training(cv,fold));
    valPatients = patients(test(cv,fold));

    trainRows = ismember(patientID,trainPatients);
    valRows = ismember(patientID,valPatients);

    XTrain = X(trainRows,:);
    YTrain = Y(trainRows);
    XVal = X(valRows,:);
    YVal = Y(valRows);

    % Fit preprocessing on the training patients only.
    imputeMedian = median(XTrain,1,'omitnan');
    imputeMedian(~isfinite(imputeMedian)) = 0;
    XTrain = imputeInvalid(XTrain,imputeMedian);
    XVal = imputeInvalid(XVal,imputeMedian);

    featureMean = mean(XTrain,1);
    featureStd = std(XTrain,0,1);
    featureStd(featureStd == 0) = 1;
    XTrain = (XTrain - featureMean) ./ featureStd;
    XVal = (XVal - featureMean) ./ featureStd;

    classCounts = countcats(YTrain);
    rawWeights = numel(YTrain) ./ (numClasses * classCounts);
    classWeights = rawWeights .^ classWeightPower;
    classWeights = classWeights ./ mean(classWeights);
    classWeights = reshape(classWeights,1,[]);
    lossFcn = @(scores,targets) crossentropy( ...
        scores,targets,classWeights,WeightsFormat="UC");

    layers = deepMLPLayers(size(XTrain,2),numClasses);
    net = trainnet(XTrain,YTrain,layers,lossFcn,options);

    scores = minibatchpredict(net,XVal,MiniBatchSize=miniBatchSize);
    predicted = string(scores2label(scores,classNames));
    actual = string(YVal);

    TP = sum(predicted == "A" & actual == "A");
    FP = sum(predicted == "A" & actual == "N");
    FN = sum(predicted == "N" & actual == "A");
    TN = sum(predicted == "N" & actual == "N");

    Sensitivity(fold) = safeDivide(TP,TP + FN);
    PPV(fold) = safeDivide(TP,TP + FP);
    F1(fold) = safeDivide(2 * Sensitivity(fold) * PPV(fold), ...
        Sensitivity(fold) + PPV(fold));
    Accuracy(fold) = safeDivide(TP + TN,TP + TN + FP + FN);

    fprintf("Fold %d: Sens = %.4f, PPV = %.4f, F1 = %.4f, Accuracy = %.4f\n", ...
        fold,Sensitivity(fold),PPV(fold),F1(fold),Accuracy(fold));
end

results = table((1:numFolds)',Sensitivity,PPV,F1,Accuracy, ...
    'VariableNames',{'Fold','Sensitivity','PPV','F1','Accuracy'});

fprintf("\n5-Fold Patient-Wise Cross Validation\n");
fprintf("Mean Sensitivity = %.4f\n",mean(Sensitivity));
fprintf("Mean PPV         = %.4f\n",mean(PPV));
fprintf("Mean F1          = %.4f\n",mean(F1));
fprintf("Mean Accuracy    = %.4f\n",mean(Accuracy));
fprintf("Std Sensitivity  = %.4f\n",std(Sensitivity));
fprintf("Std PPV          = %.4f\n",std(PPV));
fprintf("Std F1           = %.4f\n\n",std(F1));

% Train the deployable model with preprocessing fitted to all available data.
imputeMedian = median(X,1,'omitnan');
imputeMedian(~isfinite(imputeMedian)) = 0;
X = imputeInvalid(X,imputeMedian);
featureMean = mean(X,1);
featureStd = std(X,0,1);
featureStd(featureStd == 0) = 1;
X = (X - featureMean) ./ featureStd;

classCounts = countcats(Y);
rawWeights = numel(Y) ./ (numClasses * classCounts);
classWeights = rawWeights .^ classWeightPower;
classWeights = classWeights ./ mean(classWeights);
classWeights = reshape(classWeights,1,[]);
lossFcn = @(scores,targets) crossentropy( ...
    scores,targets,classWeights,WeightsFormat="UC");

layers = deepMLPLayers(size(X,2),numClasses);
[classifier, trainingInfo] = trainnet(X,Y,layers,lossFcn,options);

model.type = "deep_mlp";
model.classifier = classifier;
model.imputeMedian = imputeMedian;
model.featureMean = featureMean;
model.featureStd = featureStd;
model.classNames = classNames;
model.classWeightPower = classWeightPower;
model.trainingInfo = trainingInfo;

if ~exist("Results","dir")
    mkdir("Results");
end
save(fullfile("Results","trained_model.mat"),"model","results","-v7.3");
fprintf("Final Deep MLP model saved to Results/trained_model.mat\n");

end

function X = imputeInvalid(X,imputeMedian)
for feature = 1:size(X,2)
    X(~isfinite(X(:,feature)),feature) = imputeMedian(feature);
end
end

function layers = deepMLPLayers(numFeatures,numClasses)
layers = [
    featureInputLayer(numFeatures,Normalization="none")

    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.25)

    fullyConnectedLayer(128)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.20)

    fullyConnectedLayer(64)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.15)

    fullyConnectedLayer(32)
    reluLayer

    fullyConnectedLayer(16)
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    ];
end

function value = safeDivide(numerator,denominator)
if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end
end
