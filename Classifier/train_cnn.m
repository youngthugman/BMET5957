function net = train_cnn(XTrain,YTrain)
%TRAIN_CNN Train a compact 50-Hz to 1-Hz sequence labeller.
% XTrain entries are 2-by-(50*nSeconds); YTrain entries retain all of the
% corresponding second-level N/A labels.

labels = [YTrain{:}];
counts = [sum(labels == 'N'), sum(labels == 'A')];
assert(all(counts > 0), 'Both N and A must occur in the training patients.');
classWeights = (sum(counts)./counts).^0.75;
classWeights = classWeights / mean(classWeights);

layers = [
    sequenceInputLayer(2,Normalization='none')
    convolution1dLayer(9,16,Padding='same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(5,Stride=5)
    convolution1dLayer(7,32,Padding='same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(5,Stride=5)
    convolution1dLayer(5,64,Padding='same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,Stride=2)
    convolution1dLayer(7,64,Padding='same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer];

% Development assertion: 2 x 3000 must remain temporal and become 2 x 60.
testNet = dlnetwork(layers);
testOutput = predict(testNet,dlarray(zeros(2,3000,'single'),'CT'));
assert(size(testOutput,1) == 2 && size(testOutput,2) == 60, ...
    'The CNN must produce 60 score pairs for a 60-second block.');

lossFcn = @(Y,T) crossentropy(Y,T,Weights=classWeights, ...
    ClassificationMode='auto');
options = trainingOptions('adam', ...
    MaxEpochs=10, ...
    MiniBatchSize=32, ...
    Shuffle='every-epoch', ...
    ExecutionEnvironment='auto', ...
    Verbose=true, ...
    Plots='none');
net = trainnet(XTrain,YTrain,layers,lossFcn,options);
end
