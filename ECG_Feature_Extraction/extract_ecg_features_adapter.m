function [features, featureNames, qrs] = extract_ecg_features_adapter(ecg, FsECG, numSeconds, config, suppliedQRS)
%EXTRACT_ECG_FEATURES_ADAPTER Reuse the project detector once, then align RR features to 1 Hz.
ecg=double(ecg(:)); expected=numSeconds*FsECG;
if abs(numel(ecg)-expected)>=FsECG
    error('BMET:ECGLength','ECG duration (%.3fs) and annotations (%ds) differ by at least one second; no truncation was performed.',numel(ecg)/FsECG,numSeconds);
end
mode=lower(string(config.ecg.qrsMode));
if mode=="existing"
    [filtered,bad]=simpleFilters(ecg,FsECG);
    [~,qrs]=pan_tompkin(filtered,FsECG,false); qrs=qrs(:);
    qrs=qrs(qrs>=1&qrs<=numel(filtered));
    qrs=alignQRS(qrs,filtered,round(0.05*FsECG));
    qrs=qrs(qrs>=1&qrs<=numel(filtered)); qrs=qrs(~bad(qrs));
    qrs=removeCloseQRS(qrs,filtered,round(0.5*FsECG));
elseif mode=="supplied"
    if isempty(suppliedQRS), error('BMET:MissingQRS','qrsMode="supplied" but this patient has no supplied QRS locations.'); end
    qrs=double(suppliedQRS(:));
else, error('BMET:QRSMode','qrsMode must be "existing" or "supplied".');
end
rr=diff(qrs)/FsECG; rt=qrs(2:end)/FsECG;
valid=rr>=config.ecg.minRR & rr<=config.ecg.maxRR; rr(~valid)=NaN;
names=["ecg_rr_current","ecg_hr_current","ecg_rr_mean","ecg_rr_std","ecg_rmssd","ecg_pnn50", ...
 "ecg_rr_min","ecg_rr_max","ecg_beat_count","ecg_rr_slope"];
features=nan(numSeconds,numel(names)); b=config.contextBeforeSeconds; a=config.contextAfterSeconds;
for t=1:numSeconds
    in=rt>=(t-1-b)&rt<=(t+a); r=rr(in); times=rt(in); good=isfinite(r);
    prior=find(rt<=t & isfinite(rr),1,'last'); if isempty(prior), current=NaN; else, current=rr(prior); end
    dr=diff(r); dr=dr(isfinite(dr));
    if nnz(good)>=2, p=polyfit(times(good),r(good),1); rslope=p(1); else, rslope=0; end
    features(t,:)=[current 60/current mean(r,'omitnan') std(r,'omitnan') sqrt(mean(dr.^2,'omitnan')) ...
        100*mean(abs(dr)>0.05,'omitnan') min(r,[],'omitnan') max(r,[],'omitnan') nnz(good) rslope];
end
featureNames=names;
end
