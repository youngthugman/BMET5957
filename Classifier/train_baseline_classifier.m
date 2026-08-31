function model = train_baseline_classifier(X,Y,config)
%TRAIN_BASELINE_CLASSIFIER Train the baseline boosted-tree classifier.
%
% Handles:
%   - char, string, cell or categorical labels
%   - NaN / Inf feature values
%   - train-only median imputation
%   - optional robust normalisation
%   - categorical N/A labels safely
%
% Positive class for evaluation should remain 'A'.

%% =========================================================
%  Check required toolbox
%  =========================================================

if ~license('test','Statistics_Toolbox')
    error(['Statistics and Machine Learning Toolbox is required for ' ...
           'templateTree() and fitcensemble(). Install/enable it in MATLAB.']);
end


%% =========================================================
%  Prepare labels
%  =========================================================

% Convert whatever format Y currently has into a clean string column.
if ischar(Y)

    % Expected case:
    % Y is an Nx1 char array such as:
    % N
    % N
    % A
    % A
    Y = string(cellstr(Y));

elseif isstring(Y)

    Y = Y(:);

elseif iscell(Y)

    Y = string(Y(:));

elseif iscategorical(Y)

    Y = string(Y(:));

else

    error("Unsupported label datatype: %s", class(Y));

end

Y = strtrim(Y(:));

% Safety check: training labels must only contain N or A.
badLabels = ~(Y == "N" | Y == "A");

if any(badLabels)
    bad = unique(Y(badLabels));
    error( ...
        "Training labels contain unexpected values: %s", ...
        strjoin(bad,", "));
end

% Convert to categorical only AFTER making labels clean strings.
Y = categorical(Y, ["N","A"]);


%% =========================================================
%  Prepare feature matrix
%  =========================================================

X = double(X);

if size(X,1) ~= numel(Y)
    error( ...
        "Feature/label mismatch: X has %d rows but Y has %d labels.", ...
        size(X,1), numel(Y));
end


%% =========================================================
%  Train-only missing-value imputation
%  =========================================================

model.imputeMedian = zeros(1,size(X,2));

for j = 1:size(X,2)

    col = X(:,j);

    good = isfinite(col);

    if any(good)
        model.imputeMedian(j) = median(col(good));
    else
        model.imputeMedian(j) = 0;
    end

    col(~good) = model.imputeMedian(j);

    X(:,j) = col;

end


%% =========================================================
%  Optional robust normalisation
%  =========================================================
%
% Tree models do not actually require normalisation, but we preserve
% support because predict_baseline_classifier may already expect
% model.centre and model.scale.

if isfield(config.classifier,'normalise') && config.classifier.normalise

    model.centre = median(X,1);

    % Compute an IQR-like scale manually, avoiding iqr()/prctile()
    % dependencies here.
    model.scale = ones(1,size(X,2));

    for j = 1:size(X,2)

        sortedCol = sort(X(:,j));

        n = numel(sortedCol);

        idx25 = max(1, round(0.25 * (n-1)) + 1);
        idx75 = max(1, round(0.75 * (n-1)) + 1);

        q25 = sortedCol(idx25);
        q75 = sortedCol(idx75);

        s = q75 - q25;

        if ~isfinite(s) || s == 0
            s = 1;
        end

        model.scale(j) = s;

    end

else

    model.centre = zeros(1,size(X,2));
    model.scale  = ones(1,size(X,2));

end

X = (X - model.centre) ./ model.scale;


%% =========================================================
%  Class balance information
%  =========================================================

numN = sum(Y == 'N');
numA = sum(Y == 'A');

fprintf( ...
    'Training classifier: %d samples [N=%d, A=%d, A prevalence=%.2f%%]\n', ...
    numel(Y), ...
    numN, ...
    numA, ...
    100*numA/numel(Y));


%% =========================================================
%  Configure tree learner
%  =========================================================

tree = templateTree( ...
    'MaxNumSplits', config.classifier.maxNumSplits);


%% =========================================================
%  Train ensemble
%  =========================================================

method = char(config.classifier.method);

fprintf( ...
    'Training %s ensemble with %d learning cycles...\n', ...
    method, ...
    config.classifier.numLearningCycles);

% Some ensemble methods accept LearnRate while others do not.
switch lower(method)

    case {'logitboost','gentleboost','adaboostm1','adaboostm2'}

        model.classifier = fitcensemble( ...
            X, ...
            Y, ...
            'Method', method, ...
            'Learners', tree, ...
            'NumLearningCycles', config.classifier.numLearningCycles, ...
            'LearnRate', config.classifier.learnRate);

    case {'rusboost'}

        model.classifier = fitcensemble( ...
            X, ...
            Y, ...
            'Method', method, ...
            'Learners', tree, ...
            'NumLearningCycles', config.classifier.numLearningCycles, ...
            'LearnRate', config.classifier.learnRate);

    otherwise

        % Generic fallback.
        model.classifier = fitcensemble( ...
            X, ...
            Y, ...
            'Method', method, ...
            'Learners', tree, ...
            'NumLearningCycles', config.classifier.numLearningCycles);

end


%% =========================================================
%  Store metadata used by prediction
%  =========================================================

model.classNames = string(model.classifier.ClassNames);

% Determine which classifier score column corresponds to Apnoea.
model.apnoeaScoreColumn = find(model.classNames == "A",1);

if isempty(model.apnoeaScoreColumn)
    error("Could not find class 'A' in trained classifier.");
end

fprintf( ...
    'Classifier trained. Class order: %s | Apnoea score column: %d\n', ...
    strjoin(model.classNames,", "), ...
    model.apnoeaScoreColumn);

end