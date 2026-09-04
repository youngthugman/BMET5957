%% Controlled patient-wise model benchmark and feature ablation
% This script is exploratory: it only fits models inside cross-validation
% folds and never replaces the final model used by the working pipeline.

clearvars;
clc;

cacheFile = fullfile("Cache","train_features.mat");
requiredVariables = ["X" "Y" "patientID" "featureNames"];
if ~isfile(cacheFile)
    error("benchmark:MissingCache","Could not find %s.",cacheFile);
end
cacheContents = whos("-file",cacheFile);
availableVariables = string({cacheContents.name});
missingVariables = setdiff(requiredVariables,availableVariables);
if ~isempty(missingVariables)
    error("benchmark:MissingData", ...
        "Missing required variable(s) in %s: %s",cacheFile,join(missingVariables,", "));
end

fprintf("Loading cached features from %s...\n",cacheFile);
load(cacheFile,"X","Y","patientID","featureNames");
if size(X,1) ~= numel(Y) || size(X,1) ~= numel(patientID)
    error("benchmark:RowMismatch", ...
        "X, Y, and patientID must contain the same number of observations.");
end

classNames = ["N" "A"];
Y = categorical(string(Y),classNames);
if any(isundefined(Y))
    error("benchmark:UnknownClass","Y must contain only the classes N and A.");
end
Y = Y(:);
patientID = patientID(:);

featureNames = string(featureNames(:))';
if numel(featureNames) ~= size(X,2)
    error("benchmark:FeatureNameMismatch", ...
        "featureNames must have one entry for each column of X.");
end
ecgColumns = startsWith(featureNames,"ecg_");
spo2Columns = startsWith(featureNames,"spo2_");
if ~any(ecgColumns) || ~any(spo2Columns)
    error("benchmark:MissingModality", ...
        "Could not find both ecg_ and spo2_ feature names.");
end

featureSetNames = ["ECG only" "SpO2 only" "ECG + SpO2"];
featureSetColumns = {ecgColumns,spo2Columns,ecgColumns | spo2Columns};
for featureSet = 1:numel(featureSetNames)
    fprintf("%-12s: %d features\n",featureSetNames(featureSet), ...
        nnz(featureSetColumns{featureSet}));
end

% Construct this partition once. Every experiment below reuses it.
patients = unique(patientID);
rng(1);
numFolds = 5;
cv = cvpartition(numel(patients),'KFold',numFolds);

modelNames = ["Deep MLP" "RUSBoost" "LogitBoost" "Bagged Trees"];
numRows = size(X,1);
numExperiments = numel(featureSetNames) * numel(modelNames);
numFoldResults = numExperiments * numFolds;

resultFeatureSet = strings(numFoldResults,1);
resultModel = strings(numFoldResults,1);
resultFold = zeros(numFoldResults,1);
TP = zeros(numFoldResults,1);
FP = zeros(numFoldResults,1);
FN = zeros(numFoldResults,1);
TN = zeros(numFoldResults,1);
Sensitivity = zeros(numFoldResults,1);
PPV = zeros(numFoldResults,1);
F1 = zeros(numFoldResults,1);
Accuracy = zeros(numFoldResults,1);
resultRow = 0;

% Logical OOF labels are compact enough to retain all twelve experiments.
% MLP A scores are kept as single precision to limit memory use.
outOfFold = repmat(struct,1,numel(featureSetNames));
for featureSet = 1:numel(featureSetNames)
    outOfFold(featureSet).FeatureSet = featureSetNames(featureSet);
    outOfFold(featureSet).ActualIsA = (Y == "A");
    outOfFold(featureSet).Fold = zeros(numRows,1,"uint8");
    for modelIndex = 1:numel(modelNames)
        fieldName = matlab.lang.makeValidName(modelNames(modelIndex));
        outOfFold(featureSet).Models.(fieldName).Model = modelNames(modelIndex);
        outOfFold(featureSet).Models.(fieldName).PredictedIsA = false(numRows,1);
        outOfFold(featureSet).Models.(fieldName).Assigned = false(numRows,1);
        if modelNames(modelIndex) == "Deep MLP"
            outOfFold(featureSet).Models.(fieldName).AScore = nan(numRows,1,"single");
        end
    end
end

miniBatchSize = 4096;
for featureSet = 1:numel(featureSetNames)
    columns = featureSetColumns{featureSet};
    fprintf("\nFeature set %d/%d: %s (%d features)\n",featureSet, ...
        numel(featureSetNames),featureSetNames(featureSet),nnz(columns));

    for fold = 1:numFolds
        trainPatients = patients(training(cv,fold));
        valPatients = patients(test(cv,fold));
        trainRows = ismember(patientID,trainPatients);
        valRows = ismember(patientID,valPatients);
        if any(trainRows & valRows)
            error("benchmark:PatientLeakage","A patient appears in both fold partitions.");
        end
        outOfFold(featureSet).Fold(valRows) = uint8(fold);

        % Only training rows contribute to imputation and standardisation.
        XTrain = X(trainRows,columns);
        XVal = X(valRows,columns);
        YTrain = Y(trainRows);
        YVal = Y(valRows);

        imputeMedian = median(XTrain,1,'omitnan');
        imputeMedian(~isfinite(imputeMedian)) = 0;
        XTrain = replaceNonfinite(XTrain,imputeMedian);
        XVal = replaceNonfinite(XVal,imputeMedian);

        featureMean = mean(XTrain,1);
        featureStd = std(XTrain,0,1);
        featureStd(featureStd == 0) = 1;
        XTrain = (XTrain - featureMean) ./ featureStd;
        XVal = (XVal - featureMean) ./ featureStd;

        for modelIndex = 1:numel(modelNames)
            modelName = modelNames(modelIndex);
            fprintf("  Model %d/%d: %s | Fold %d/%d...\n",modelIndex, ...
                numel(modelNames),modelName,fold,numFolds);

            if modelName == "Deep MLP"
                [predicted,AScore] = fitAndPredictMLP( ...
                    XTrain,YTrain,XVal,classNames,miniBatchSize);
            else
                predicted = fitAndPredictEnsemble( ...
                    XTrain,YTrain,XVal,classNames,modelName);
                AScore = [];
            end

            predictedIsA = string(predicted(:)) == "A";
            actualIsA = (YVal == "A");
            metrics = classificationMetrics(predictedIsA,actualIsA);

            resultRow = resultRow + 1;
            resultFeatureSet(resultRow) = featureSetNames(featureSet);
            resultModel(resultRow) = modelName;
            resultFold(resultRow) = fold;
            TP(resultRow) = metrics.TP;
            FP(resultRow) = metrics.FP;
            FN(resultRow) = metrics.FN;
            TN(resultRow) = metrics.TN;
            Sensitivity(resultRow) = metrics.Sensitivity;
            PPV(resultRow) = metrics.PPV;
            F1(resultRow) = metrics.F1;
            Accuracy(resultRow) = metrics.Accuracy;

            fieldName = matlab.lang.makeValidName(modelName);
            outOfFold(featureSet).Models.(fieldName).PredictedIsA(valRows) = predictedIsA;
            outOfFold(featureSet).Models.(fieldName).Assigned(valRows) = true;
            if modelName == "Deep MLP"
                outOfFold(featureSet).Models.(fieldName).AScore(valRows) = single(AScore);
            end

            fprintf("    %s | %s | Fold %d: Sens=%.4f PPV=%.4f F1=%.4f Acc=%.4f\n", ...
                featureSetNames(featureSet),modelName,fold,metrics.Sensitivity, ...
                metrics.PPV,metrics.F1,metrics.Accuracy);
            clear predicted AScore
        end
        clear XTrain XVal YTrain YVal
    end
end

foldResults = table(resultFeatureSet,resultModel,resultFold,TP,FP,FN,TN, ...
    Sensitivity,PPV,F1,Accuracy,'VariableNames', ...
    {'FeatureSet','Model','Fold','TP','FP','FN','TN','Sensitivity','PPV','F1','Accuracy'});

% Confirm that each row received exactly one held-out result per experiment.
for featureSet = 1:numel(featureSetNames)
    if any(outOfFold(featureSet).Fold == 0)
        error("benchmark:IncompleteOOF","Some rows were not assigned to a fold.");
    end
    for modelIndex = 1:numel(modelNames)
        fieldName = matlab.lang.makeValidName(modelNames(modelIndex));
        if ~all(outOfFold(featureSet).Models.(fieldName).Assigned)
            error("benchmark:IncompleteOOF","Some held-out predictions are missing.");
        end
    end
end

%% Exploratory out-of-fold threshold analysis for the Deep MLP
thresholds = (0.30:0.02:0.80)';
thresholdResults = table;
mlpField = matlab.lang.makeValidName("Deep MLP");
fprintf("\nDeep MLP out-of-fold threshold analysis\n");
for featureSet = 1:numel(featureSetNames)
    scores = double(outOfFold(featureSet).Models.(mlpField).AScore);
    actualIsA = outOfFold(featureSet).ActualIsA;
    n = numel(thresholds);
    thresholdFeatureSet = repmat(featureSetNames(featureSet),n,1);
    thresholdSensitivity = zeros(n,1);
    thresholdPPV = zeros(n,1);
    thresholdF1 = zeros(n,1);
    thresholdAccuracy = zeros(n,1);
    for thresholdIndex = 1:n
        metrics = classificationMetrics(scores >= thresholds(thresholdIndex),actualIsA);
        thresholdSensitivity(thresholdIndex) = metrics.Sensitivity;
        thresholdPPV(thresholdIndex) = metrics.PPV;
        thresholdF1(thresholdIndex) = metrics.F1;
        thresholdAccuracy(thresholdIndex) = metrics.Accuracy;
    end
    thisTable = table(thresholdFeatureSet,thresholds,thresholdSensitivity, ...
        thresholdPPV,thresholdF1,thresholdAccuracy,'VariableNames', ...
        {'FeatureSet','Threshold','Sensitivity','PPV','F1','Accuracy'});
    thresholdResults = [thresholdResults; thisTable]; %#ok<AGROW>

    fprintf("\n%s:\n",featureSetNames(featureSet));
    reportThresholdChoice(thisTable,"Default threshold",thisTable.Threshold == 0.50,"F1");
    reportThresholdChoice(thisTable,"Maximum F1",true(height(thisTable),1),"F1");
    eligible = thisTable.Sensitivity >= 0.80;
    reportThresholdChoice(thisTable,"Maximum F1 with Sens >= 0.80",eligible,"F1");
    reportThresholdChoice(thisTable,"Maximum PPV with Sens >= 0.80",eligible,"PPV");
end

%% Fold-mean summary and numerical feature-ablation comparison
summaryFeatureSet = strings(numExperiments,1);
summaryModel = strings(numExperiments,1);
MeanSensitivity = zeros(numExperiments,1); StdSensitivity = zeros(numExperiments,1);
MeanPPV = zeros(numExperiments,1); StdPPV = zeros(numExperiments,1);
MeanF1 = zeros(numExperiments,1); StdF1 = zeros(numExperiments,1);
MeanAccuracy = zeros(numExperiments,1); StdAccuracy = zeros(numExperiments,1);
row = 0;
for featureSet = 1:numel(featureSetNames)
    for modelIndex = 1:numel(modelNames)
        row = row + 1;
        useRows = foldResults.FeatureSet == featureSetNames(featureSet) & ...
            foldResults.Model == modelNames(modelIndex);
        summaryFeatureSet(row) = featureSetNames(featureSet);
        summaryModel(row) = modelNames(modelIndex);
        MeanSensitivity(row) = mean(foldResults.Sensitivity(useRows));
        StdSensitivity(row) = std(foldResults.Sensitivity(useRows));
        MeanPPV(row) = mean(foldResults.PPV(useRows));
        StdPPV(row) = std(foldResults.PPV(useRows));
        MeanF1(row) = mean(foldResults.F1(useRows));
        StdF1(row) = std(foldResults.F1(useRows));
        MeanAccuracy(row) = mean(foldResults.Accuracy(useRows));
        StdAccuracy(row) = std(foldResults.Accuracy(useRows));
    end
end
summaryTable = table(summaryFeatureSet,summaryModel,MeanSensitivity, ...
    StdSensitivity,MeanPPV,StdPPV,MeanF1,StdF1,MeanAccuracy,StdAccuracy, ...
    'VariableNames',{'FeatureSet','Model','MeanSensitivity','StdSensitivity', ...
    'MeanPPV','StdPPV','MeanF1','StdF1','MeanAccuracy','StdAccuracy'});
summaryTable = sortrows(summaryTable,"MeanF1","descend");

fprintf("\nOverall ranking by mean fold F1\n");
disp(summaryTable);
eligibleSummary = summaryTable(summaryTable.MeanSensitivity >= 0.80,:);
fprintf("\nRanking by mean F1, restricted to mean sensitivity >= 0.80\n");
if isempty(eligibleSummary)
    fprintf("No experiment met the mean sensitivity constraint.\n");
else
    disp(eligibleSummary);
end

fprintf("\nNumerical feature-ablation comparison (mean fold F1)\n");
for modelIndex = 1:numel(modelNames)
    modelRows = summaryTable(summaryTable.Model == modelNames(modelIndex),:);
    ecgF1 = modelRows.MeanF1(modelRows.FeatureSet == "ECG only");
    spo2F1 = modelRows.MeanF1(modelRows.FeatureSet == "SpO2 only");
    combinedF1 = modelRows.MeanF1(modelRows.FeatureSet == "ECG + SpO2");
    fprintf("%s: ECG=%.4f, SpO2=%.4f, Combined=%.4f\n", ...
        modelNames(modelIndex),ecgF1,spo2F1,combinedF1);
    fprintf("  Adding SpO2 to ECG: %+.4f; adding ECG to SpO2: %+.4f\n", ...
        combinedF1-ecgF1,combinedF1-spo2F1);
    if ecgF1 < spo2F1
        fprintf("  ECG has the lower single-modality F1 in this benchmark.\n");
    elseif spo2F1 < ecgF1
        fprintf("  SpO2 has the lower single-modality F1 in this benchmark.\n");
    else
        fprintf("  The single-modality F1 values are equal in this benchmark.\n");
    end
end

%% Save benchmark outputs only (never the submission model)
if ~exist("Results","dir")
    mkdir("Results");
end
save(fullfile("Results","benchmark_results.mat"), ...
    "summaryTable","foldResults","thresholdResults","outOfFold", ...
    "featureNames","classNames","-v7.3");
writetable(summaryTable,fullfile("Results","benchmark_summary.csv"));
writetable(thresholdResults,fullfile("Results","benchmark_thresholds.csv"));
fprintf("\nSaved benchmark results in Results/benchmark_results.mat and CSV summaries.\n");

%% Local helper functions
function XData = replaceNonfinite(XData,medians)
for feature = 1:size(XData,2)
    bad = ~isfinite(XData(:,feature));
    XData(bad,feature) = medians(feature);
end
end

function [predicted,AScore] = fitAndPredictMLP( ...
        XTrain,YTrain,XVal,classNames,miniBatchSize)
numFeatures = size(XTrain,2);
numClasses = numel(classNames);
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

classCounts = countcats(YTrain);
if any(classCounts == 0)
    error("benchmark:MissingTrainingClass","Both N and A are required in every training fold.");
end
rawWeights = numel(YTrain) ./ (numClasses * classCounts);
classWeights = rawWeights .^ 0.75;
classWeights = classWeights ./ mean(classWeights);
classWeights = reshape(classWeights,1,[]);
lossFcn = @(scores,targets) crossentropy( ...
    scores,targets,classWeights,WeightsFormat="UC");

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

% Deliberately no ValidationData: outer-fold patients are evaluated afterward.
net = trainnet(XTrain,YTrain,layers,lossFcn,options);
scores = minibatchpredict(net,XVal,MiniBatchSize=miniBatchSize);
scores = gather(scores);
predicted = scores2label(scores,classNames);
aColumn = find(classNames == "A",1);
AScore = scores(:,aColumn);
clear net
end

function predicted = fitAndPredictEnsemble( ...
        XTrain,YTrain,XVal,classNames,modelName)
tree = templateTree(MaxNumSplits=100,MinLeafSize=50);
switch modelName
    case "RUSBoost"
        method = "RUSBoost";
    case "LogitBoost"
        method = "LogitBoost";
    case "Bagged Trees"
        method = "Bag";
    otherwise
        error("benchmark:UnknownModel","Unknown model: %s",modelName);
end
model = fitcensemble(XTrain,YTrain,Method=method,Learners=tree, ...
    NumLearningCycles=100,ClassNames=categorical(classNames,classNames));
predicted = predict(model,XVal);
clear model
end

function metrics = classificationMetrics(predictedIsA,actualIsA)
metrics.TP = sum(predictedIsA & actualIsA);
metrics.FP = sum(predictedIsA & ~actualIsA);
metrics.FN = sum(~predictedIsA & actualIsA);
metrics.TN = sum(~predictedIsA & ~actualIsA);
metrics.Sensitivity = safeDivide(metrics.TP,metrics.TP + metrics.FN);
metrics.PPV = safeDivide(metrics.TP,metrics.TP + metrics.FP);
metrics.F1 = safeDivide(2 * metrics.Sensitivity * metrics.PPV, ...
    metrics.Sensitivity + metrics.PPV);
metrics.Accuracy = safeDivide(metrics.TP + metrics.TN, ...
    metrics.TP + metrics.FP + metrics.FN + metrics.TN);
end

function value = safeDivide(numerator,denominator)
if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end
end

function reportThresholdChoice(results,label,eligible,criterion)
rows = find(eligible);
if isempty(rows)
    fprintf("  %s: no eligible threshold\n",label);
    return
end
if numel(rows) > 1
    [~,bestWithinRows] = max(results.(criterion)(rows));
    row = rows(bestWithinRows);
else
    row = rows;
end
fprintf("  %s: t=%.2f Sens=%.4f PPV=%.4f F1=%.4f Acc=%.4f\n", ...
    label,results.Threshold(row),results.Sensitivity(row),results.PPV(row), ...
    results.F1(row),results.Accuracy(row));
end
