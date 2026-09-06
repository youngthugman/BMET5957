function [model, results] = train_classifier(X, Y, patientID)

classNames = ["N" "A"];
Y = categorical(string(Y),classNames);
if any(isundefined(Y))
    error("Y must contain only the class labels N and A.");
end

patients = unique(patientID);
rng(1);
numFolds = 5;
cv = cvpartition(numel(patients),'KFold',numFolds);

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

    % Use only training-fold statistics to fill invalid values.
    imputeMedian = median(XTrain,1,'omitnan');
    imputeMedian(~isfinite(imputeMedian)) = 0;
    for feature = 1:size(XTrain,2)
        XTrain(~isfinite(XTrain(:,feature)),feature) = imputeMedian(feature);
        XVal(~isfinite(XVal(:,feature)),feature) = imputeMedian(feature);
    end

    tree = templateTree( ...
        MaxNumSplits=200, ...
        MinLeafSize=50);
    classifier = fitcensemble( ...
        XTrain,YTrain, ...
        Method="RUSBoost", ...
        Learners=tree, ...
        NumLearningCycles=200, ...
        LearnRate=0.05);

    predicted = string(predict(classifier,XVal));
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
fprintf("Mean Accuracy    = %.4f\n",mean(Accuracy));
fprintf("Std Sensitivity  = %.4f\n",std(Sensitivity));
fprintf("Std PPV          = %.4f\n",std(PPV));
fprintf("Std F1           = %.4f\n\n",std(F1));

% Train the final model on all patients using all-data medians.
imputeMedian = median(X,1,'omitnan');
imputeMedian(~isfinite(imputeMedian)) = 0;
for feature = 1:size(X,2)
    X(~isfinite(X(:,feature)),feature) = imputeMedian(feature);
end

tree = templateTree( ...
    MaxNumSplits=200, ...
    MinLeafSize=50);
classifier = fitcensemble( ...
    X,Y, ...
    Method="RUSBoost", ...
    Learners=tree, ...
    NumLearningCycles=200, ...
    LearnRate=0.05);

model.classifier = classifier;
model.imputeMedian = imputeMedian;
model.classNames = classNames;
model.type = "rusboost";

if ~exist("Results","dir")
    mkdir("Results");
end
save(fullfile("Results","trained_model.mat"),"model","results","-v7.3");
fprintf("Final RUSBoost model saved to Results/trained_model.mat\n");

end
