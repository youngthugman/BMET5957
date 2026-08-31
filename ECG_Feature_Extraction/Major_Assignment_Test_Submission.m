clear
clc

% BMET3997 test submission script
% Mirrors the Major_Assignment processing pipeline, with optional
% sanity-check plotting for test records.
% Pipeline:
% ECG -> simple preprocessing -> Pan-Tompkins -> QRS cleanup -> HRV -> save .mat

load ProjectTestData.mat

fs = 100;
gr = false;

groupNumber = 2;
submissionNumber = 5;     % change this if needed

% Optional sanity-check plotting (set [] to disable plotting)
plotRecords = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35];
plotSamples = 500000;

nRecords = length(ECG);

QRS = cell(1, nRecords);

avgRR = nan(1, nRecords);          % seconds internally
sdRR = nan(1, nRecords);           % seconds internally
RMSSD = nan(1, nRecords);          % seconds internally
pNN50 = nan(1, nRecords);          % percent
LF = nan(1, nRecords);             % seconds^2 internally
HF = nan(1, nRecords);             % seconds^2 internally
LF_HFratio = nan(1, nRecords);     % unitless

for i = 1:nRecords
    fprintf('Processing test record %d / %d...\n', i, nRecords);

    signal = ECG{i}(:);

    % 1. Preprocess ECG
    [signalFiltered, badSegment] = simpleFilters(signal, fs);

    % 2. First-stage QRS detection
    [~, qrs] = pan_tompkin(signalFiltered, fs, gr);
    qrs = qrs(:);

    % 3. Fiducial-point refinement
    qrs = qrs(qrs >= 1 & qrs <= length(signalFiltered));
    qrs = alignQRS(qrs, signalFiltered, 5);

    % 4. Post-detection refinement
    qrs = qrs(qrs >= 1 & qrs <= length(signalFiltered));
    qrs = qrs(~badSegment(qrs));
    qrs = removeCloseQRS(qrs, signalFiltered, round(0.500 * fs));

    QRS{i} = qrs;

    % 5. HRV calculation
    [avgRR(i), sdRR(i), RMSSD(i), pNN50(i), LF(i), HF(i), LF_HFratio(i)] = ...
        calculateHRV(qrs, length(signal), fs);

    fprintf('Record %02d: QRS = %d, avgRR = %.4f s\n', ...
        i, length(qrs), avgRR(i));

    % Optional sanity-check plot: filtered test ECG and predicted QRS
    if ismember(i, plotRecords)
        plotEnd = min(plotSamples, length(signalFiltered));
        x = 1:plotEnd;
        qrsPlot = qrs(qrs >= 1 & qrs <= plotEnd);

        figure;
        plot(x, signalFiltered(x), 'b-');
        hold on;
        plot(qrsPlot, signalFiltered(qrsPlot), 'r*');
        xlabel('Sample');
        ylabel('ECG amplitude');
        title(['Test Record ', num2str(i), ': Filtered ECG with Predicted QRS']);
        legend('Filtered ECG', 'Predicted QRS');
    end
end

% Quick sanity check before unit conversion
qrsCounts = cellfun(@length, QRS);

fprintf('\nTest-data sanity summary:\n');
fprintf('QRS count: min = %d, median = %.0f, max = %d\n', ...
    min(qrsCounts), median(qrsCounts), max(qrsCounts));

fprintf('NaN counts before saving:\n');
fprintf('  avgRR      = %d\n', sum(isnan(avgRR)));
fprintf('  sdRR       = %d\n', sum(isnan(sdRR)));
fprintf('  RMSSD      = %d\n', sum(isnan(RMSSD)));
fprintf('  pNN50      = %d\n', sum(isnan(pNN50)));
fprintf('  LF         = %d\n', sum(isnan(LF)));
fprintf('  HF         = %d\n', sum(isnan(HF)));
fprintf('  LF/HF      = %d\n', sum(isnan(LF_HFratio)));

% Convert to required submission units
avgRR = avgRR * 1000;          % seconds to ms
sdRR = sdRR * 1000;            % seconds to ms
RMSSD = RMSSD * 1000;          % seconds to ms

LF = LF * 1000000;             % seconds^2 to ms^2
HF = HF * 1000000;             % seconds^2 to ms^2

% pNN50 is already percent
% LF_HFratio is already unitless

% Save only required submission variables
outputFileName = sprintf('ProjectTestDataAnalysisGroup%dSubmission%d.mat', ...
    groupNumber, submissionNumber);

save(outputFileName, ...
    'QRS', ...
    'avgRR', ...
    'sdRR', ...
    'RMSSD', ...
    'pNN50', ...
    'LF', ...
    'HF', ...
    'LF_HFratio');

fprintf('\nSaved submission file: %s\n', outputFileName);

fprintf('\nVariables saved in submission file:\n');
whos('-file', outputFileName)
