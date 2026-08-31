function y = preprocess_spo2(x, config)
%PREPROCESS_SPO2 Apply optional light, non-destructive smoothing.
y = double(x(:));
switch lower(string(config.preprocessing))
    case "none"
    case "median", y = movmedian(y,config.medianWidth,'omitnan','Endpoints','shrink');
    case "gaussian", y = smoothdata(y,'gaussian',config.gaussianWidth,'omitnan');
    otherwise, error('BMET:SpO2Preprocessing','Unknown SpO2 preprocessing mode: %s',config.preprocessing);
end
end
