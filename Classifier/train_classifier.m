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
oofScores = nan(size(Y));
oofActual = strings(size(Y));
oofCount = zeros(size(Y));
aColumn = find(classNames == "A");

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
    aScores = gather(scores(:,aColumn));
    predictedA = aScores >= 0.50;
    actual = string(YVal);
    oofScores(valRows) = aScores;
    oofActual(valRows) = actual;
    oofCount(valRows) = oofCount(valRows) + 1;

    TP = sum(predictedA & actual == "A");
    FP = sum(predictedA & actual == "N");
    FN = sum(~predictedA & actual == "A");
    TN = sum(~predictedA & actual == "N");

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

if any(oofCount ~= 1) || any(isnan(oofScores)) || any(oofActual == "")
    error("Every sample must have exactly one out-of-fold prediction.");
end

% Threshold selection is based on OOF cross-validation predictions.
% Final external evaluation is still required on the hidden test set.
thresholds = (0.30:0.01:0.80)';
numThresholds = numel(thresholds);
thresholdSensitivity = zeros(numThresholds,1);
thresholdPPV = zeros(numThresholds,1);
thresholdF1 = zeros(numThresholds,1);
thresholdAccuracy = zeros(numThresholds,1);
for index = 1:numThresholds
    [thresholdSensitivity(index),thresholdPPV(index),thresholdF1(index), ...
        thresholdAccuracy(index)] = classificationMetrics( ...
        oofScores >= thresholds(index),oofActual);
end
thresholdResults = table(thresholds,thresholdSensitivity,thresholdPPV, ...
    thresholdF1,thresholdAccuracy,'VariableNames', ...
    {'Threshold','Sensitivity','PPV','F1','Accuracy'});

[~,bestF1Index] = max(thresholdF1);
eligible = find(thresholdSensitivity >= 0.80);
if isempty(eligible)
    bestConstrainedF1Index = [];
    bestConstrainedPPVIndex = [];
    selectedIndex = bestF1Index;
else
    [~,relativeIndex] = max(thresholdF1(eligible));
    bestConstrainedF1Index = eligible(relativeIndex);
    [~,relativeIndex] = max(thresholdPPV(eligible));
    bestConstrainedPPVIndex = eligible(relativeIndex);
    selectedIndex = bestConstrainedF1Index;
end
selectedThreshold = thresholds(selectedIndex);

printThreshold("Best F1 threshold",thresholdResults(bestF1Index,:));
if isempty(bestConstrainedF1Index)
    fprintf("\nNo threshold satisfies Sensitivity >= 0.80.\n");
else
    printThreshold("Best F1 threshold with Sensitivity >= 0.80", ...
        thresholdResults(bestConstrainedF1Index,:));
    printThreshold("Best PPV threshold with Sensitivity >= 0.80", ...
        thresholdResults(bestConstrainedPPVIndex,:));
end

[defaultSensitivity,defaultPPV,defaultF1] = classificationMetrics( ...
    oofScores >= 0.50,oofActual);
fprintf("\nDefault threshold 0.50:\n");
fprintf("  Sens = %.4f\n  PPV  = %.4f\n  F1   = %.4f\n", ...
    defaultSensitivity,defaultPPV,defaultF1);
fprintf("\nSelected threshold:\n");
fprintf("  Threshold = %.2f\n  Sens = %.4f\n  PPV  = %.4f\n  F1   = %.4f\n\n", ...
    selectedThreshold,thresholdSensitivity(selectedIndex), ...
    thresholdPPV(selectedIndex),thresholdF1(selectedIndex));

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
model.threshold = selectedThreshold;

if ~exist("Results","dir")
    mkdir("Results");
end
save(fullfile("Results","trained_model.mat"),"model","results", ...
    "thresholdResults","selectedThreshold","-v7.3");
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

function [sensitivity,ppv,f1,accuracy] = classificationMetrics(predictedA,actual)
TP = sum(predictedA & actual == "A");
FP = sum(predictedA & actual == "N");
FN = sum(~predictedA & actual == "A");
TN = sum(~predictedA & actual == "N");
sensitivity = safeDivide(TP,TP + FN);
ppv = safeDivide(TP,TP + FP);
f1 = safeDivide(2 * sensitivity * ppv,sensitivity + ppv);
accuracy = safeDivide(TP + TN,TP + TN + FP + FN);
end

function printThreshold(titleText,row)
fprintf("\n%s:\n",titleText);
fprintf("Threshold = %.2f\n",row.Threshold);
fprintf("Sens = %.4f\nPPV  = %.4f\nF1   = %.4f\nAcc  = %.4f\n", ...
    row.Sensitivity,row.PPV,row.F1,row.Accuracy);
end
