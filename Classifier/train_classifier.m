function [model, results] = train_classifier(X, Y, patientID)

Y = categorical(cellstr(Y));                 % Convert N/A labels to classes
patients = unique(patientID);                 % List of unique patients

numFolds = 5;
rng(1);                                       % Same CV split each run
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

    % Fill missing values using TRAINING medians only
    imputeMedian = median(XTrain,1,'omitnan');
    imputeMedian(~isfinite(imputeMedian)) = 0;

    for feature = 1:size(XTrain,2)
        bad = ~isfinite(XTrain(:,feature));
        XTrain(bad,feature) = imputeMedian(feature);

        bad = ~isfinite(XVal(:,feature));
        XVal(bad,feature) = imputeMedian(feature);
    end

    % Standardise using TRAINING mean and std only
    featureMean = mean(XTrain,1);
    featureStd = std(XTrain,0,1);
    featureStd(featureStd == 0) = 1;

    XTrain = (XTrain - featureMean) ./ featureStd;
    XVal = (XVal - featureMean) ./ featureStd;

    % Train temporary SVM for this fold
    classifier = fitclinear(XTrain,YTrain,'Learner','svm');

    % Predict patients not seen during training
    predicted = predict(classifier,XVal);

    predicted = string(predicted);
    actual = string(YVal);

    TP = sum(predicted == "A" & actual == "A");
    FP = sum(predicted == "A" & actual == "N");
    FN = sum(predicted == "N" & actual == "A");
    TN = sum(predicted == "N" & actual == "N");

    Sensitivity(fold) = TP / (TP + FN);

    if TP + FP > 0
        PPV(fold) = TP / (TP + FP);
    end

    if Sensitivity(fold) + PPV(fold) > 0
        F1(fold) = 2 * Sensitivity(fold) * PPV(fold) / ...
            (Sensitivity(fold) + PPV(fold));
    end

    Accuracy(fold) = (TP + TN) / (TP + TN + FP + FN);

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

classifier = fitclinear(X,Y,'Learner','svm');

model.classifier = classifier;
model.imputeMedian = imputeMedian;
model.featureMean = featureMean;
model.featureStd = featureStd;

fprintf("Final SVM trained using all patients.\n");

end