function cacheFiles = build_feature_cache(dataPath, templatePath, split, config)
%BUILD_FEATURE_CACHE Extract and cache one patient at a time.
isTrain=split=="train"; sourceInfo=validate_dataset(dataPath,split);
if isTrain, templateInfo=sourceInfo; else, templateInfo=validate_dataset(templatePath,"template"); end
if sourceInfo.numPatients~=templateInfo.numPatients, error('BMET:PatientCount','Signal and annotation patient counts differ.'); end
cacheDir=fullfile(config.outputDirectory,'FeatureCache'); if ~isfolder(cacheDir), mkdir(cacheDir); end
signature=feature_config_signature(config); cacheFiles=strings(sourceInfo.numPatients,1);
for i=1:sourceInfo.numPatients
    cacheFile=fullfile(cacheDir,sprintf('%s_patient_%03d.mat',split,i)); cacheFiles(i)=cacheFile;
    if isfile(cacheFile) && ~config.forceFeatureExtraction
        c=load(cacheFile,'extractionSignature');
        if isfield(c,'extractionSignature') && strcmp(c.extractionSignature,signature), fprintf('%s patient %d/%d: cache reused\n',split,i,sourceInfo.numPatients); continue; end
    end
    r=load_patient_record(dataPath,i,isTrain);
    if isTrain, labels=char(r.Class); else, tm=matfile(templatePath); z=tm.Class(1,i); labels=char(z{1}); end
    annotationLength=numel(labels);
    fprintf('%s patient %d/%d: ECG %d samples, %d seconds\n',split,i,sourceInfo.numPatients,numel(r.ECG),annotationLength);
    [ECGFeatures,ECGFeatureNames]=extract_ecg_features_adapter(r.ECG,config.ecg.sampleRate,annotationLength,config,r.QRS);
    [SpO2Features,SpO2FeatureNames]=extract_spo2_features(r.SpO2,annotationLength,config);
    [combinedFeatures,featureNames]=combine_features(ECGFeatures,ECGFeatureNames,SpO2Features,SpO2FeatureNames,annotationLength);
    extractionSignature=signature; %#ok<NASGU>
    save(cacheFile,'ECGFeatures','SpO2Features','combinedFeatures','featureNames','annotationLength','labels','extractionSignature','-v7.3');
    fprintf('  ECG %dx%d, SpO2 %dx%d, combined %dx%d; cache saved\n',size(ECGFeatures),size(SpO2Features),size(combinedFeatures));
end
end
