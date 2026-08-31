function [features, featureNames] = extract_spo2_features(spo2, numSeconds, config)
%EXTRACT_SPO2_FEATURES One offline, centred feature row per annotation second.
if numel(spo2) ~= numSeconds
    error('BMET:SpO2Length','SpO2 has %d samples but annotations have %d seconds; no truncation was performed.',numel(spo2),numSeconds);
end
x = preprocess_spo2(spo2,config.spo2);
widths = config.spo2.windowHalfWidths(:)';
if numel(widths)==1 && widths==20
    before=config.contextBeforeSeconds; after=config.contextAfterSeconds;
    scales=[before after];
else
    scales=[widths(:) widths(:)];
end
baseNames = ["current","mean","median","std","min","max","range","p10","p90", ...
 "drop_from_max","drop_from_median","drop_from_prebaseline","max_drop_5s","max_drop_10s","max_drop_20s", ...
 "future_min_from_current","future_min_from_prebaseline","slope_full","slope_pre","slope_post", ...
 "derivative_min","derivative_max","derivative_mean","derivative_mean_abs", ...
 "delta_past_5s","delta_past_10s","delta_past_20s","delta_future_5s","delta_future_10s","delta_future_20s", ...
 "desaturation_area","time_to_min","time_since_max","recovery_slope","count_drop3","count_drop4"];
features = nan(numSeconds,numel(baseNames)*size(scales,1)); featureNames=strings(1,size(features,2));
for s=1:size(scales,1)
    b=scales(s,1); a=scales(s,2); cols=(s-1)*numel(baseNames)+(1:numel(baseNames));
    featureNames(cols)="spo2_b"+b+"_a"+a+"_"+baseNames;
    for t=1:numSeconds
        lo=max(1,t-b); hi=min(numSeconds,t+a); w=x(lo:hi); rel=(lo:hi)'-t;
        pre=x(lo:t); post=x(t:hi); cur=x(t); d=diff(w);
        prebase=median(pre,'omitnan'); [mn,imin]=min(w,[],'omitnan'); [~,imax]=max(w,[],'omitnan');
        drops=arrayfun(@(h) maxDrop(x,max(1,t-h),min(numSeconds,t+h),h),[5 10 20]);
        dp=arrayfun(@(h) cur-x(max(1,t-h)),[5 10 20]);
        df=arrayfun(@(h) x(min(numSeconds,t+h))-cur,[5 10 20]);
        baseline=max(prebase,cur); area=sum(max(0,baseline-w),'omitnan');
        values=[cur mean(w,'omitnan') median(w,'omitnan') std(w,'omitnan') mn max(w,[],'omitnan') range(w) ...
            prctile(w,10) prctile(w,90) max(w,[],'omitnan')-cur median(w,'omitnan')-cur prebase-cur drops ...
            min(post,[],'omitnan')-cur min(post,[],'omitnan')-prebase slope(rel,w) slope((lo:t)'-t,pre) slope((t:hi)'-t,post) ...
            min(d,[],'omitnan') max(d,[],'omitnan') mean(d,'omitnan') mean(abs(d),'omitnan') dp df area rel(imin) -rel(imax) ...
            slope((t:hi)'-t,post) countDrops(w,3) countDrops(w,4)];
        features(t,cols)=values;
    end
end
end
function v=slope(t,x), good=isfinite(t)&isfinite(x); if nnz(good)<2, v=0; else, p=polyfit(t(good),x(good),1); v=p(1); end, end
function v=maxDrop(x,lo,hi,h), v=0; for k=lo:hi, v=max(v,x(k)-min(x(k:min(hi,k+h)),[],'omitnan')); end, end
function n=countDrops(x,threshold), n=0; peak=x(1); active=false; for k=2:numel(x), peak=max(peak,x(k)); if peak-x(k)>=threshold && ~active, n=n+1; active=true; elseif x(k)>=peak-1, peak=x(k); active=false; end, end, end
