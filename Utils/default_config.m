function config = default_config(outputDirectory)
%DEFAULT_CONFIG Return the reproducible version-1 pipeline configuration.
config.randomSeed = 42;
config.contextBeforeSeconds = 20;
config.contextAfterSeconds = 20;
config.forceFeatureExtraction = false;
config.runCrossValidation = true;
config.trainFinalModel = true;
config.predictTestData = true;
config.outputDirectory = outputDirectory;
config.ecg.sampleRate = 200;
config.ecg.qrsMode = "existing"; % existing detector is the default; "supplied" is a benchmark
config.ecg.minRR = 0.30;
config.ecg.maxRR = 2.00;
config.spo2.sampleRate = 1;
config.spo2.preprocessing = "median";
config.spo2.medianWidth = 3;
config.spo2.gaussianWidth = 5;
config.spo2.windowHalfWidths = 20;
config.classifier.method = "RUSBoost";
config.classifier.numLearningCycles = 150;
config.classifier.maxNumSplits = 20;
config.classifier.learnRate = 0.1;
config.classifier.normalise = true;
config.crossValidation.numFolds = 5;
config.thresholdGrid = 0.10:0.01:0.90;
end
