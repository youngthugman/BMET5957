function net = train_tcn(XTrain,YTrain,classWeights,maxEpochs)
%TRAIN_TCN Train a non-causal sequence-to-sequence feature TCN.
% Stored observations are T-by-26 and targets are T-by-1 categorical.
% All temporal convolutions use symmetric "same" padding.

layers = tcnLayers();
net = dlnetwork(layers);

% Calling with no inputs only constructs the network (used by shape tests).
if nargin == 0
    return
end

if nargin < 4
    maxEpochs = 20;
end

assert(iscell(XTrain) && iscell(YTrain) && numel(XTrain) == numel(YTrain), ...
    "Inputs and targets must be equally sized cell arrays.");
assert(all(cellfun(@(x) size(x,2) == 26,XTrain)), ...
    "Every input sequence must be T-by-26.");
assert(all(cellfun(@(y) size(y,2) == 1,YTrain)), ...
    "Every target sequence must be T-by-1.");
assert(all(cellfun(@(x,y) size(x,1) == size(y,1),XTrain,YTrain)), ...
    "Every input and target sequence must have the same length.");
assert(all(cellfun(@(x) size(x,1) >= 127,XTrain)), ...
    "Every sequence must contain at least the 127-second MinLength.");
assert(all(cellfun(@(y) isequal(string(categories(y)),["N";"A"]),YTrain)), ...
    "Every target must use categorical class order [N; A].");
assert(all(isfinite(classWeights)) && numel(classWeights) == 2, ...
    "classWeights must contain finite weights in [N A] order.");

% One datastore observation is one complete, variable-length patient.
inputStore = arrayDatastore(XTrain,OutputType="same");
targetStore = arrayDatastore(YTrain,OutputType="same");
trainingData = combine(inputStore,targetStore);

classWeights = single(reshape(classWeights,1,2));
lossFcn = @(scores,targets) 2 * crossentropy( ...
    scores,targets,classWeights, ...
    WeightsFormat="UC", ...
    ClassificationMode="single-label", ...
    NormalizationFactor="all-elements");

options = trainingOptions("adam", ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=1, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="auto", ...
    L2Regularization=1e-4, ...
    InputDataFormats="TCB", ...
    TargetDataFormats="TCB", ...
    Verbose=true, ...
    Plots="none");

net = trainnet(trainingData,net,lossFcn,options);
end


function layers = tcnLayers()
% Six kernel-3 stages give 1 + 2*sum([1 2 4 8 16 32]) = 127 s.
layers = [
    sequenceInputLayer(26,Normalization="none",MinLength=127,Name="features")
    convolution1dLayer(1,64,Padding="same",Name="input_projection")
    layerNormalizationLayer(Name="input_norm")
    reluLayer(Name="input_relu")

    temporalStage(1,"d1")
    temporalStage(2,"d2")
    temporalStage(4,"d4")
    temporalStage(8,"d8")
    temporalStage(16,"d16")
    temporalStage(32,"d32")

    convolution1dLayer(1,2,Padding="same",Name="class_scores")
    softmaxLayer(Name="probabilities")
    ];
end


function layers = temporalStage(dilation,name)
layers = [
    convolution1dLayer(3,64,Padding="same",DilationFactor=dilation, ...
        Name=name + "_conv")
    layerNormalizationLayer(Name=name + "_norm")
    reluLayer(Name=name + "_relu")
    dropoutLayer(0.10,Name=name + "_dropout")
    ];
end
