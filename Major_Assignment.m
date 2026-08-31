%hello

clear
clc
% ECG -> simple preprocessing -> Pan-Tompkins -> QRS cleanup -> HRV -> metrics
load ProjectTrainData.mat
fs = 100;
gr = false;
tol = round(0.050 * fs);          % 50 ms = 5 samples at 100 Hz
plotRecords = 25;         % records to plot, set [] to disable [17 25 33]
plotSamples = 500000;              % number of samples to show in plots
nRecords = length(ECG);
QRS = cell(1, nRecords);
avgRR = nan(1, nRecords);
sdRR = nan(1, nRecords);
RMSSD = nan(1, nRecords);
pNN50 = nan(1, nRecords);
LF = nan(1, nRecords);
HF = nan(1, nRecords);
LF_HFratio = nan(1, nRecords);
TP = zeros(1, nRecords);
FP = zeros(1, nRecords);
FN = zeros(1, nRecords);
for i = 1:nRecords
    fprintf('Processing record %d / %d...\n', i, nRecords);
    signal = ECG{i}(:);
    % 1. Preprocess ECG
    [signalFiltered, badSegment] = simpleFilters(signal, fs);
    % 2. First-stage QRS detection
    [~, qrs] = pan_tompkin(signalFiltered, fs, gr);
    qrs = qrs(:);
    
    % 3. Fiducial-point refinement
    % Pan-Tompkins finds the QRS region. This moves each detection to the
    % strongest local ECG peak nearby.
    qrs = qrs(qrs >= 1 & qrs <= length(signalFiltered));
    qrs = alignQRS(qrs, signalFiltered, 5); % tried 7 and it dropped f1 to 0.9917
    
    % 4. Post-detection refinement
    qrs = qrs(qrs >= 1 & qrs <= length(signalFiltered));
    qrs = qrs(~badSegment(qrs));
    qrs = removeCloseQRS(qrs, signalFiltered, round(0.500 * fs)); %was 0.250
    QRS{i} = qrs;
    % 4. HRV calculation
    [avgRR(i), sdRR(i), RMSSD(i), pNN50(i), LF(i), HF(i), LF_HFratio(i)] = ...
        calculateHRV(qrs, length(signal), fs);
    % 5. Training metrics
    [TP(i), FP(i), FN(i)] = qrsMetrics(qrs, QRSexpert{i}, tol);
    sens_i = TP(i) / (TP(i) + FN(i));
    ppv_i = TP(i) / (TP(i) + FP(i));
    f1_i = 2 * sens_i * ppv_i / (sens_i + ppv_i);
    fprintf('Record %02d: QRS = %d, Sens = %.4f, PPV = %.4f, F1 = %.4f, avgRR = %.4f s\n', ...
        i, length(qrs), sens_i, ppv_i, f1_i, avgRR(i));
    % Plot our QRS detections vs expert detections for selected records
    if ismember(i, plotRecords)
        plotEnd = min(plotSamples, length(signalFiltered));
        x = 1:plotEnd;
        qrsPlot = qrs(qrs >= 1 & qrs <= plotEnd);
        expertPlot = QRSexpert{i};
        expertPlot = expertPlot(expertPlot >= 1 & expertPlot <= plotEnd);
        figure;
        plot(x, signalFiltered(x));
        hold on;
        plot(qrsPlot, signalFiltered(qrsPlot), 'r*');
        plot(expertPlot, signalFiltered(expertPlot), 'go');
        xlabel('Sample');
        ylabel('ECG amplitude');
        title(['Record ', num2str(i), ': Our QRS vs Expert QRS']);
        legend('Filtered ECG', 'Our QRS', 'Expert QRS');
    end
end
% Final QRS metrics
Sensitivity = sum(TP) / (sum(TP) + sum(FN));
PPV = sum(TP) / (sum(TP) + sum(FP));
F1 = 2 * Sensitivity * PPV / (Sensitivity + PPV);
fprintf('\nFinal QRS Training Results:\n');
fprintf('Sensitivity = %.4f\n', Sensitivity);
fprintf('PPV         = %.4f\n', PPV);
fprintf('F1-score    = %.4f\n', F1);
% avgRR MAPE
if ~exist('avgRRexpert', 'var')
    avgRRexpert = [
        0.9924 0.7498 0.9218 0.9599 0.9487 0.9904 0.8195 ...
        0.7310 0.9472 0.9631 0.8496 0.9765 0.7480 1.0855 ...
        0.9053 0.8328 0.7880 0.9782 0.7762 0.8888 0.8359 ...
        0.9018 0.9129 1.0324 0.9430 1.0307 0.9221 1.1506 ...
        0.9900 0.9991 1.0009 0.8094 1.0217 0.8799 1.0535
    ];
end
avgRRexpert = avgRRexpert(:).';
MAPE_avgRR = mean(abs((avgRR - avgRRexpert) ./ avgRRexpert) * 100, 'omitnan');
fprintf('\nTraining avgRR MAPE = %.4f %%\n', MAPE_avgRR);
% -------------------------------------------------------------------------
% Expert-QRS-derived HRV benchmark
% -------------------------------------------------------------------------
% This uses QRSexpert to calculate reference HRV values with the same
% calculateHRV() function. This is not the hidden staff HRV exactly, but it
% is a useful gauge of how much HRV error comes from our QRS detections.
avgRR_ref = nan(1, nRecords);
sdRR_ref = nan(1, nRecords);
RMSSD_ref = nan(1, nRecords);
pNN50_ref = nan(1, nRecords);
LF_ref = nan(1, nRecords);
HF_ref = nan(1, nRecords);
LF_HFratio_ref = nan(1, nRecords);
for i = 1:nRecords
    [avgRR_ref(i), sdRR_ref(i), RMSSD_ref(i), pNN50_ref(i), LF_ref(i), HF_ref(i), LF_HFratio_ref(i)] = ...
        calculateHRV(QRSexpert{i}, length(ECG{i}), fs);
end
MAPE_avgRR_ref = mean(abs((avgRR - avgRR_ref) ./ avgRR_ref) * 100, 'omitnan');
MAPE_sdRR_ref = mean(abs((sdRR - sdRR_ref) ./ sdRR_ref) * 100, 'omitnan');
MAPE_RMSSD_ref = mean(abs((RMSSD - RMSSD_ref) ./ RMSSD_ref) * 100, 'omitnan');
MAPE_pNN50_ref = mean(abs((pNN50 - pNN50_ref) ./ pNN50_ref) * 100, 'omitnan');
MAPE_LF_ref = mean(abs((LF - LF_ref) ./ LF_ref) * 100, 'omitnan');
MAPE_HF_ref = mean(abs((HF - HF_ref) ./ HF_ref) * 100, 'omitnan');
MAPE_LF_HFratio_ref = mean(abs((LF_HFratio - LF_HFratio_ref) ./ LF_HFratio_ref) * 100, 'omitnan');
averageMAPE_ref = mean([ ...
    MAPE_avgRR_ref, ...
    MAPE_sdRR_ref, ...
    MAPE_RMSSD_ref, ...
    MAPE_pNN50_ref, ...
    MAPE_LF_ref, ...
    MAPE_HF_ref, ...
    MAPE_LF_HFratio_ref ...
], 'omitnan');
fprintf('\nExpert-QRS-derived HRV MAPE:\n');
fprintf('MAPE_avgRR      = %.4f %%\n', MAPE_avgRR_ref);
fprintf('MAPE_sdRR       = %.4f %%\n', MAPE_sdRR_ref);
fprintf('MAPE_RMSSD      = %.4f %%\n', MAPE_RMSSD_ref);
fprintf('MAPE_pNN50      = %.4f %%\n', MAPE_pNN50_ref);
fprintf('MAPE_LF         = %.4f %%\n', MAPE_LF_ref);
fprintf('MAPE_HF         = %.4f %%\n', MAPE_HF_ref);
fprintf('MAPE_LF_HFratio = %.4f %%\n', MAPE_LF_HFratio_ref);
fprintf('averageMAPE     = %.4f %%\n', averageMAPE_ref);
HRVcompare = table((1:nRecords)', ...
    avgRR(:), avgRR_ref(:), ...
    sdRR(:), sdRR_ref(:), ...
    RMSSD(:), RMSSD_ref(:), ...
    pNN50(:), pNN50_ref(:), ...
    LF(:), LF_ref(:), ...
    HF(:), HF_ref(:), ...
    LF_HFratio(:), LF_HFratio_ref(:), ...
    'VariableNames', { ...
    'Record', ...
    'avgRR_ours', 'avgRR_ref', ...
    'sdRR_ours', 'sdRR_ref', ...
    'RMSSD_ours', 'RMSSD_ref', ...
    'pNN50_ours', 'pNN50_ref', ...
    'LF_ours', 'LF_ref', ...
    'HF_ours', 'HF_ref', ...
    'LF_HFratio_ours', 'LF_HFratio_ref'});
disp(HRVcompare);
% Simple HRV summary
fprintf('\nHRV summary, internal units:\n');
fprintf('avgRR, sdRR, RMSSD are seconds. LF and HF are seconds^2.\n\n');
disp(table((1:nRecords)', avgRR(:), sdRR(:), RMSSD(:), pNN50(:), LF(:), HF(:), LF_HFratio(:), ...
    'VariableNames', {'Record', 'avgRR_s', 'sdRR_s', 'RMSSD_s', 'pNN50_percent', 'LF_s2', 'HF_s2', 'LF_HFratio'}));
