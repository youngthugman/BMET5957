%% USER CONFIGURATION
% Edit only these portable paths for a first run.
trainDataPath = fullfile(pwd,"ProjectTrainData.mat");
testDataPath = fullfile(pwd,"ProjectTestData.mat");
testAnnotationTemplatePath = fullfile(pwd,"ProjectTestAnnotations.mat");
outputDirectory = fullfile(pwd,"Results");

%% Configuration
addpath(genpath(pwd));
config = default_config(outputDirectory);
% Examples: config.ecg.qrsMode="supplied"; config.spo2.preprocessing="none";
% Multi-scale example: config.spo2.windowHalfWidths=[10 20 30 60];
rng(config.randomSeed);
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

%% Load / validate data
trainInfo=validate_dataset(trainDataPath,"train");
if isfield(trainInfo,'SR_ECG') && trainInfo.SR_ECG~=config.ecg.sampleRate, error('Configured ECG rate disagrees with SR_ECG.'); end
if isfield(trainInfo,'SR_SpO2') && trainInfo.SR_SpO2~=config.spo2.sampleRate, error('Configured SpO2 rate disagrees with SR_SpO2.'); end

%% ECG feature extraction / SpO2 feature extraction / Build feature dataset
trainCache=build_feature_cache(trainDataPath,"","train",config);
[X,Y,patientID,featureNames]=load_cached_dataset(trainCache,true);

%% Patient-wise cross-validation / Evaluate CV / Optimise threshold
if config.runCrossValidation
    cv=patientwise_cross_validation(X,Y,patientID,config);
    writetable(cv.foldMetrics,fullfile(outputDirectory,'cv_metrics.csv'));
    save(fullfile(outputDirectory,'cv_results.mat'),'cv','config','featureNames','-v7.3');
    chosenThreshold=cv.threshold;
else
    warning('Cross-validation disabled: threshold 0.5 has not been optimised on held-out patients.'); chosenThreshold=0.5;
end

%% Train final model
if config.trainFinalModel
    trainedModel=train_baseline_classifier(X,Y,config);
    metadata.created=datetime('now','TimeZone','UTC'); metadata.positiveClass='A'; metadata.trainingPatients=numel(unique(patientID));
    save(fullfile(outputDirectory,'trained_model.mat'),'trainedModel','featureNames','config','chosenThreshold','metadata','-v7.3');
end

%% Predict test set / Generate submission MAT file
if config.predictTestData
    if ~exist('trainedModel','var'), z=load(fullfile(outputDirectory,'trained_model.mat')); trainedModel=z.trainedModel; chosenThreshold=z.chosenThreshold; end
    testCache=build_feature_cache(testDataPath,testAnnotationTemplatePath,"test",config);
    [XTest,~,~,testFeatureNames,testLengths]=load_cached_dataset(testCache,false);
    assert(isequal(string(featureNames),string(testFeatureNames)),'Training/test feature schemas differ.');
    [testPredictions,testScores]=predict_baseline_classifier(trainedModel,XTest,chosenThreshold); %#ok<NASGU>
    predictionsByPatient=cell(size(testLengths)); cursor=1;
    for i=1:numel(testLengths), predictionsByPatient{i}=testPredictions(cursor:cursor+testLengths(i)-1); cursor=cursor+testLengths(i); end
    write_test_annotations(testAnnotationTemplatePath,predictionsByPatient,fullfile(outputDirectory,'ProjectTestAnnotationsPredicted.mat'));
    save(fullfile(outputDirectory,'test_scores.mat'),'testScores','testLengths','-v7.3');
end

%% Save results / Final summary
fprintf('\n==================================================\nBMET5934 BASELINE RESULTS\n==================================================\n');
if exist('cv','var')
    m=cv.pooledMetrics; fprintf('CV Sensitivity: %.3f\nCV PPV:         %.3f\nCV F1:          %.3f\nCV Specificity: %.3f\nCV Accuracy:    %.3f\n\n',m.Sensitivity,m.PPV,m.F1,m.Specificity,m.Accuracy);
end
fprintf('Optimal threshold: %.3f\nTotal features: %d\nECG context: existing whole-record detector; centred RR summaries\nSpO2 window: -%d to +%d sec\nClassifier: %s\n==================================================\n', ...
    chosenThreshold,numel(featureNames),config.contextBeforeSeconds,config.contextAfterSeconds,config.classifier.method);
