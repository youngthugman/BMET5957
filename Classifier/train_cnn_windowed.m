function net = train_cnn_windowed(XECG,XSpO2,Y)
%TRAIN_CNN_WINDOWED Build or train the dual-branch, one-window/one-label CNN.
% Stored observations are ECG 550-by-1 and SpO2 11-by-1 (time-by-channel).

layers = windowedNetwork();
net = dlnetwork(layers);

% A no-input call supports the required architecture check before CV starts.
if nargin == 0
    return
end

assert(size(XECG,1) == 550 && size(XECG,2) == 1,"ECG must be 550-by-1-by-N.");
assert(size(XSpO2,1) == 11 && size(XSpO2,2) == 1,"SpO2 must be 11-by-1-by-N.");
assert(size(XECG,3) == size(XSpO2,3) && size(XECG,3) == numel(Y), ...
    "Inputs and scalar targets must have equal observation counts.");
assert(isequal(string(categories(Y)),["N";"A"]),"Class order must be [N, A].");

% Derive weights only from windows selected from the training patients.
counts = [sum(Y == "N"),sum(Y == "A")];
assert(all(counts > 0),"Both N and A must occur in the training patients.");
rawWeights = sum(counts)./counts;
classWeights = rawWeights.^0.75;
classWeights = reshape(classWeights/mean(classWeights),1,[]);
lossFcn = @(scores,targets) crossentropy(scores,targets,classWeights, ...
    WeightsFormat="UC",ClassificationMode="single-label");

ecgStore = arrayDatastore(XECG,IterationDimension=3);
spo2Store = arrayDatastore(XSpO2,IterationDimension=3);
labelStore = arrayDatastore(Y,IterationDimension=1);
trainingData = combine(ecgStore,spo2Store,labelStore);

options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=512, ...
    InitialLearnRate=1e-3, ...
    ExecutionEnvironment="auto", ...
    Shuffle="every-epoch", ...
    InputDataFormats=["TCB","TCB"], ...
    Verbose=true, ...
    Plots="none");

net = trainnet(trainingData,net,lossFcn,options);
end

function net = windowedNetwork()
ecgBranch = [
    sequenceInputLayer(1,Normalization="none",MinLength=550,Name="ecg")
    convolution1dLayer(101,16,Padding="same",Stride=2,Name="ecg_conv1")
    batchNormalizationLayer(Name="ecg_bn1")
    reluLayer(Name="ecg_relu1")
    maxPooling1dLayer(5,Stride=5,Name="ecg_pool1")
    convolution1dLayer(11,32,Padding="same",Name="ecg_conv2")
    batchNormalizationLayer(Name="ecg_bn2")
    reluLayer(Name="ecg_relu2")
    maxPooling1dLayer(5,Stride=5,Name="ecg_pool2")
    convolution1dLayer(7,64,Padding="same",Name="ecg_conv3")
    batchNormalizationLayer(Name="ecg_bn3")
    reluLayer(Name="ecg_relu3")
    flattenLayer(Name="ecg_flatten")];

spo2Branch = [
    sequenceInputLayer(1,Normalization="none",MinLength=11,Name="spo2")
    convolution1dLayer(5,8,Padding="same",Name="spo2_conv1")
    reluLayer(Name="spo2_relu1")
    convolution1dLayer(5,16,Padding="same",Name="spo2_conv2")
    reluLayer(Name="spo2_relu2")
    convolution1dLayer(3,16,Padding="same",Name="spo2_conv3")
    reluLayer(Name="spo2_relu3")
    flattenLayer(Name="spo2_flatten")];

fusion = [
    concatenationLayer(1,2,Name="fusion")
    fullyConnectedLayer(128,Name="fc1")
    reluLayer(Name="fusion_relu1")
    dropoutLayer(0.25,Name="dropout")
    fullyConnectedLayer(32,Name="fc2")
    reluLayer(Name="fusion_relu2")
    fullyConnectedLayer(2,Name="scores")
    softmaxLayer(Name="probabilities")];

net = layerGraph();
net = addLayers(net,ecgBranch);
net = addLayers(net,spo2Branch);
net = addLayers(net,fusion);
net = connectLayers(net,"ecg_flatten","fusion/in1");
net = connectLayers(net,"spo2_flatten","fusion/in2");
end
