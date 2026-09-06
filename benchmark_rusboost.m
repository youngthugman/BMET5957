%% Benchmark RUSBoost using cached ECG and SpO2 features
% This script evaluates models only. It does not train or save a final model.

clear;
clc;

cacheFile = fullfile("Cache", "train_features.mat");
if ~isfile(cacheFile)
    error("benchmark_rusboost:MissingCache", ...
        ["Cache/train_features.mat was not found. Run the feature extraction " ...
         "workflow to generate train_features.mat before running this benchmark."]);
end

cached = load(cacheFile);
requiredVariables = ["X", "Y", "patientID", "featureNames"];
missingVariables = requiredVariables(~isfield(cached, cellstr(requiredVariables)));
if ~isempty(missingVariables)
    error("benchmark_rusboost:IncompleteCache", ...
        "Cache/train_features.mat is missing required variable(s): %s", ...
        strjoin(missingVariables, ", "));
end

X = cached.X;
Y = cached.Y;
patientID = cached.patientID;
featureNames = string(cached.featureNames(:))';
clear cached;

if size(X, 1) ~= numel(Y) || size(X, 1) ~= numel(patientID)
    error("benchmark_rusboost:InvalidCache", ...
        "X, Y, and patientID must contain the same number of observations.");
end
if size(X, 2) ~= numel(featureNames)
    error("benchmark_rusboost:InvalidCache", ...
        "The number of featureNames must equal the number of columns in X.");
end

classNames = ["N" "A"];
Y = categorical(string(Y(:)), classNames, classNames);
if any(isundefined(Y))
    error("benchmark_rusboost:InvalidLabels", ...
        "Y must contain only the class labels N and A.");
end
patientID = patientID(:);

ecgColumns = startsWith(featureNames, "ecg_");
spo2Columns = startsWith(featureNames, "spo2_");
featureSetNames = ["ECG only"; "SpO2 only"; "ECG + SpO2"];
featureMasks = {ecgColumns, spo2Columns, ecgColumns | spo2Columns};

fprintf("Feature counts:\n");
for featureSetIndex = 1:numel(featureSetNames)
    fprintf("  %-12s = %d\n", featureSetNames(featureSetIndex), ...
        nnz(featureMasks{featureSetIndex}));
    if ~any(featureMasks{featureSetIndex})
        error("benchmark_rusboost:MissingFeatures", ...
            "No columns were found for feature set '%s'.", ...
            featureSetNames(featureSetIndex));
    end
end

% Rows are: NumLearningCycles, MaxNumSplits, MinLeafSize, LearnRate.
configurationValues = [ ...
    100, 100,  50, 0.10; ...
    200, 100,  50, 0.10; ...
    200, 200,  50, 0.05; ...
    300, 200, 100, 0.05];
numConfigurations = size(configurationValues, 1);

patients = unique(patientID);
rng(1);
numFolds = 5;
cv = cvpartition(numel(patients), "KFold", numFolds);

resultsDirectory = "Results";
if ~isfolder(resultsDirectory)
    mkdir(resultsDirectory);
end
checkpointFile = fullfile(resultsDirectory, "rusboost_checkpoint.mat");

foldResults = makeEmptyFoldResults(featureSetNames, configurationValues, numFolds);
if isfile(checkpointFile)
    checkpoint = load(checkpointFile);
    checkpointFields = ["foldResults", "featureSetNames", ...
        "configurationValues", "numFolds", "patientList"];
    if ~all(isfield(checkpoint, cellstr(checkpointFields)))
        error("benchmark_rusboost:InvalidCheckpoint", ...
            "The RUSBoost checkpoint is incomplete and cannot be resumed safely.");
    end
    if ~isequal(checkpoint.featureSetNames, featureSetNames) || ...
            ~isequal(checkpoint.configurationValues, configurationValues) || ...
            ~isequal(checkpoint.numFolds, numFolds) || ...
            ~isequaln(checkpoint.patientList, patients) || ...
            height(checkpoint.foldResults) ~= height(foldResults)
        error("benchmark_rusboost:CheckpointMismatch", ...
            ["The checkpoint does not match the current patients or benchmark " ...
             "settings. Move or rename it before starting a different benchmark."]);
    end
    foldResults = checkpoint.foldResults;
    fprintf("Checkpoint found.\nResuming benchmark...\n");
end

for featureSetIndex = 1:numel(featureSetNames)
    selectedColumns = featureMasks{featureSetIndex};
    for configurationIndex = 1:numConfigurations
        for fold = 1:numFolds
            resultRow = resultRowNumber(featureSetIndex, configurationIndex, ...
                fold, numConfigurations, numFolds);
            if foldResults.Complete(resultRow)
                fprintf("Skipping %s | Config %d | Fold %d (already complete)\n", ...
                    featureSetNames(featureSetIndex), configurationIndex, fold);
                continue;
            end

            trainPatients = patients(training(cv, fold));
            valPatients = patients(test(cv, fold));
            assert(isempty(intersect(trainPatients, valPatients)), ...
                "Patient leakage detected between training and validation sets.");

            trainRows = ismember(patientID, trainPatients);
            valRows = ismember(patientID, valPatients);
            XTrain = X(trainRows, selectedColumns);
            XVal = X(valRows, selectedColumns);
            YTrain = Y(trainRows);
            YVal = Y(valRows);

            imputeMedian = median(XTrain, 1, "omitnan");
            imputeMedian(~isfinite(imputeMedian)) = 0;
            XTrain = applyMedianImputation(XTrain, imputeMedian);
            XVal = applyMedianImputation(XVal, imputeMedian);

            tree = templateTree( ...
                MaxNumSplits=configurationValues(configurationIndex, 2), ...
                MinLeafSize=configurationValues(configurationIndex, 3));
            classifier = fitcensemble(XTrain, YTrain, ...
                Method="RUSBoost", ...
                Learners=tree, ...
                NumLearningCycles=configurationValues(configurationIndex, 1), ...
                LearnRate=configurationValues(configurationIndex, 4));

            predictedY = predict(classifier, XVal);
            actualPositive = string(YVal) == "A";
            predictedPositive = string(predictedY) == "A";
            TP = sum(actualPositive & predictedPositive);
            FP = sum(~actualPositive & predictedPositive);
            FN = sum(actualPositive & ~predictedPositive);
            TN = sum(~actualPositive & ~predictedPositive);

            sensitivity = safeDivide(TP, TP + FN);
            ppv = safeDivide(TP, TP + FP);
            f1 = safeDivide(2 * sensitivity * ppv, sensitivity + ppv);
            accuracy = safeDivide(TP + TN, TP + TN + FP + FN);

            foldResults.TP(resultRow) = TP;
            foldResults.FP(resultRow) = FP;
            foldResults.FN(resultRow) = FN;
            foldResults.TN(resultRow) = TN;
            foldResults.Sensitivity(resultRow) = sensitivity;
            foldResults.PPV(resultRow) = ppv;
            foldResults.F1(resultRow) = f1;
            foldResults.Accuracy(resultRow) = accuracy;
            foldResults.Complete(resultRow) = true;

            fprintf("%s | Config %d | Fold %d:\n", ...
                featureSetNames(featureSetIndex), configurationIndex, fold);
            fprintf("Sens=%.4f PPV=%.4f F1=%.4f Acc=%.4f\n", ...
                sensitivity, ppv, f1, accuracy);

            % Save after every fold so an interrupted run can resume here.
            patientList = patients; %#ok<NASGU>
            save(checkpointFile, "foldResults", "featureSetNames", ...
                "configurationValues", "numFolds", "patientList", "classNames");
            clear classifier tree predictedY XTrain XVal YTrain YVal;
        end
    end
end

if ~all(foldResults.Complete)
    error("benchmark_rusboost:IncompleteBenchmark", ...
        "Not all fold experiments completed; the checkpoint has been retained.");
end

summaryTable = makeSummaryTable(foldResults, featureSetNames, ...
    configurationValues, numFolds);
summaryTable = sortrows(summaryTable, "MeanF1", "descend");

fprintf("\n=========================================\n");
fprintf("RUSBOOST BENCHMARK SUMMARY\n");
fprintf("=========================================\n");
disp(summaryTable);

bestOverall = summaryTable(1, :);
printSelectedResult("Best overall configuration by MeanF1", bestOverall);

sensitivityEligible = summaryTable.MeanSensitivity >= 0.80;
if any(sensitivityEligible)
    eligibleByF1 = sortrows(summaryTable(sensitivityEligible, :), ...
        "MeanF1", "descend");
    printSelectedResult("Best MeanF1 with MeanSensitivity >= 0.80", ...
        eligibleByF1(1, :));
    eligibleByPPV = sortrows(summaryTable(sensitivityEligible, :), ...
        "MeanPPV", "descend");
    printSelectedResult("Best MeanPPV with MeanSensitivity >= 0.80", ...
        eligibleByPPV(1, :));
else
    fprintf("\nNo configuration achieved MeanSensitivity >= 0.80.\n");
end

bestByFeatureSet = summaryTable([], :);
for featureSetIndex = 1:numel(featureSetNames)
    candidates = summaryTable(summaryTable.FeatureSet == ...
        featureSetNames(featureSetIndex), :);
    bestByFeatureSet = [bestByFeatureSet; candidates(1, :)]; %#ok<AGROW>
    heading = "BEST " + upper(featureSetNames(featureSetIndex)) + " RUSBOOST";
    printSelectedResult(heading, candidates(1, :));
end

fprintf("\nFeature Set       Sensitivity    PPV       F1       Accuracy\n");
fprintf("-------------------------------------------------------------\n");
for row = 1:height(bestByFeatureSet)
    fprintf("%-17s %-14.4f %-9.4f %-8.4f %.4f\n", ...
        bestByFeatureSet.FeatureSet(row), ...
        bestByFeatureSet.MeanSensitivity(row), bestByFeatureSet.MeanPPV(row), ...
        bestByFeatureSet.MeanF1(row), bestByFeatureSet.MeanAccuracy(row));
end

ecgBestF1 = bestByFeatureSet.MeanF1(bestByFeatureSet.FeatureSet == "ECG only");
spo2BestF1 = bestByFeatureSet.MeanF1(bestByFeatureSet.FeatureSet == "SpO2 only");
combinedBestF1 = bestByFeatureSet.MeanF1( ...
    bestByFeatureSet.FeatureSet == "ECG + SpO2");
fprintf("\nAdding SpO2 to ECG changed Mean F1 by: %+.4f\n", ...
    combinedBestF1 - ecgBestF1);
fprintf("Adding ECG to SpO2 changed Mean F1 by: %+.4f\n", ...
    combinedBestF1 - spo2BestF1);

save(fullfile(resultsDirectory, "rusboost_benchmark.mat"), ...
    "foldResults", "summaryTable", "bestOverall", "bestByFeatureSet", ...
    "featureSetNames", "configurationValues", "classNames");
writetable(summaryTable, fullfile(resultsDirectory, "rusboost_summary.csv"));
fprintf("\nBenchmark results saved in Results/. Checkpoint retained.\n");

%% Local helper functions
function foldResults = makeEmptyFoldResults(featureSetNames, configurations, numFolds)
numConfigurations = size(configurations, 1);
numRows = numel(featureSetNames) * numConfigurations * numFolds;
featureSet = strings(numRows, 1);
configuration = zeros(numRows, 1);
fold = zeros(numRows, 1);

for featureSetIndex = 1:numel(featureSetNames)
    for configurationIndex = 1:numConfigurations
        for foldIndex = 1:numFolds
            row = resultRowNumber(featureSetIndex, configurationIndex, ...
                foldIndex, numConfigurations, numFolds);
            featureSet(row) = featureSetNames(featureSetIndex);
            configuration(row) = configurationIndex;
            fold(row) = foldIndex;
        end
    end
end

nanColumn = nan(numRows, 1);
foldResults = table(featureSet, configuration, fold, nanColumn, nanColumn, ...
    nanColumn, nanColumn, nanColumn, nanColumn, nanColumn, nanColumn, ...
    false(numRows, 1), VariableNames={"FeatureSet", "Configuration", "Fold", ...
    "TP", "FP", "FN", "TN", "Sensitivity", "PPV", "F1", "Accuracy", ...
    "Complete"});
end

function row = resultRowNumber(featureSetIndex, configurationIndex, fold, ...
        numConfigurations, numFolds)
row = (featureSetIndex - 1) * numConfigurations * numFolds + ...
    (configurationIndex - 1) * numFolds + fold;
end

function values = applyMedianImputation(values, imputeMedian)
for column = 1:size(values, 2)
    invalidRows = ~isfinite(values(:, column));
    values(invalidRows, column) = imputeMedian(column);
end
end

function value = safeDivide(numerator, denominator)
if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end
end

function summaryTable = makeSummaryTable(foldResults, featureSetNames, ...
        configurations, numFolds)
numConfigurations = size(configurations, 1);
numRows = numel(featureSetNames) * numConfigurations;
featureSet = strings(numRows, 1);
configuration = zeros(numRows, 1);
statistics = zeros(numRows, 8);
row = 0;

for featureSetIndex = 1:numel(featureSetNames)
    for configurationIndex = 1:numConfigurations
        row = row + 1;
        selected = foldResults.FeatureSet == featureSetNames(featureSetIndex) & ...
            foldResults.Configuration == configurationIndex;
        assert(nnz(selected) == numFolds && all(foldResults.Complete(selected)), ...
            "Summary requested for an incomplete experiment.");
        featureSet(row) = featureSetNames(featureSetIndex);
        configuration(row) = configurationIndex;
        statistics(row, :) = [ ...
            mean(foldResults.Sensitivity(selected)), std(foldResults.Sensitivity(selected)), ...
            mean(foldResults.PPV(selected)), std(foldResults.PPV(selected)), ...
            mean(foldResults.F1(selected)), std(foldResults.F1(selected)), ...
            mean(foldResults.Accuracy(selected)), std(foldResults.Accuracy(selected))];
    end
end

summaryTable = table(featureSet, configuration, ...
    configurations(configuration, 1), configurations(configuration, 2), ...
    configurations(configuration, 3), configurations(configuration, 4), ...
    statistics(:, 1), statistics(:, 2), statistics(:, 3), statistics(:, 4), ...
    statistics(:, 5), statistics(:, 6), statistics(:, 7), statistics(:, 8), ...
    VariableNames={"FeatureSet", "Configuration", "NumLearningCycles", ...
    "MaxNumSplits", "MinLeafSize", "LearnRate", "MeanSensitivity", ...
    "StdSensitivity", "MeanPPV", "StdPPV", "MeanF1", "StdF1", ...
    "MeanAccuracy", "StdAccuracy"});
end

function printSelectedResult(titleText, result)
fprintf("\n%s\n", titleText);
fprintf("  %s | Configuration %d\n", result.FeatureSet, result.Configuration);
fprintf("  Mean Sensitivity: %.4f\n", result.MeanSensitivity);
fprintf("  Mean PPV:         %.4f\n", result.MeanPPV);
fprintf("  Mean F1:          %.4f\n", result.MeanF1);
fprintf("  Mean Accuracy:    %.4f\n", result.MeanAccuracy);
end
