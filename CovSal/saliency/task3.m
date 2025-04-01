%% Clear workspace
clear all;
clc;

% Add path to the cvzoya/saliency metrics functions
% (Assuming these functions are in a directory called 'saliency_metrics')
% addpath('saliency_metrics');

fprintf('Starting saliency map evaluation...\n');

% Define folders
td_fixation_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\TD_FixMaps';
asd_fixation_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\ASD_FixMaps';
prediction_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency_Maps';

fprintf('Loading prediction files...\n');
% Get list of prediction files (assuming .png files)
prediction_files = dir(fullfile(prediction_folder, 'Saliency_*.png'));
num_images = length(prediction_files);

fprintf('Initializing metric storage...\n');
% Initialize metric storage (8 metrics)
metrics_td = zeros(num_images, 8);
metrics_asd = zeros(num_images, 8);

fprintf('Processing images...\n');
% Loop through each image
for i = 1:num_images
    % Extract image number from prediction file name
    file_name = prediction_files(i).name;
    start_idx = find(file_name == '_') + 1;
    end_idx = find(file_name == '.') - 1;
    image_num = str2double(file_name(start_idx:end_idx));
    
    fprintf('Processing image %d of %d (Image number: %d)\n', i, num_images, image_num);
    
    % Load prediction map
    prediction_path = fullfile(prediction_folder, file_name);
    prediction_map = imread(prediction_path);
    prediction_map = im2double(prediction_map);
    
    % Load TD fixation map
    td_fixation_file = sprintf('%d_s.png', image_num);
    td_fixation_path = fullfile(td_fixation_folder, td_fixation_file);
    td_fixation_map = imread(td_fixation_path);
    td_fixation_map = im2double(td_fixation_map);
    
    % Load ASD fixation map
    asd_fixation_file = sprintf('%d_s.png', image_num);
    asd_fixation_path = fullfile(asd_fixation_folder, asd_fixation_file);
    asd_fixation_map = imread(asd_fixation_path);
    asd_fixation_map = im2double(asd_fixation_map);
    
    % Compute metrics for TD
    fprintf('  Computing metrics for TD...\n');
    metrics_td(i,:) = compute_all_metrics(prediction_map, td_fixation_map);
    
    % Compute metrics for ASD
    fprintf('  Computing metrics for ASD...\n');
    metrics_asd(i,:) = compute_all_metrics(prediction_map, asd_fixation_map);
    
    fprintf('  Image %d processed successfully\n\n', image_num);
end

fprintf('Computing mean metrics...\n');
% Compute mean metrics
mean_metrics_td = mean(metrics_td);
mean_metrics_asd = mean(metrics_asd);

fprintf('Displaying results...\n');
% Display results
fprintf('\nEvaluation Results:\n');
fprintf('Metric\t\t\tTD\t\tASD\n');
fprintf('AUC_Borji\t\t%.4f\t%.4f\n', mean_metrics_td(1), mean_metrics_asd(1));
fprintf('AUC_Judd\t\t%.4f\t%.4f\n', mean_metrics_td(2), mean_metrics_asd(2));
fprintf('AUC_shuffled\t\t%.4f\t%.4f\n', mean_metrics_td(3), mean_metrics_asd(3));
fprintf('CC\t\t\t%.4f\t%.4f\n', mean_metrics_td(4), mean_metrics_asd(4));
fprintf('EMD\t\t\t%.4f\t%.4f\n', mean_metrics_td(5), mean_metrics_asd(5));
fprintf('Info Gain\t\t%.4f\t%.4f\n', mean_metrics_td(6), mean_metrics_asd(6));
fprintf('KLdiv\t\t\t%.4f\t%.4f\n', mean_metrics_td(7), mean_metrics_asd(7));
fprintf('NSS\t\t\t%.4f\t%.4f\n', mean_metrics_td(8), mean_metrics_asd(8));

% Function to compute all evaluation metrics
function metrics = compute_all_metrics(saliency_map, fixation_map)
    fprintf('    Entering compute_all_metrics function...\n');
    
    fprintf('    Normalizing saliency map...\n');
    % Normalize saliency map
    saliency_map = (saliency_map - min(saliency_map(:))) / (max(saliency_map(:)) - min(saliency_map(:)));
    
    fprintf('    Creating binary fixation map...\n');
    % Create binary fixation map
    fixation_map_binary = fixation_map > 0;
    
    fprintf('    Creating continuous fixation map...\n');
    % Create continuous fixation map by blurring the binary map
    fixation_map_continuous = antonioGaussian(double(fixation_map_binary), 8);
    fixation_map_continuous = fixation_map_continuous / max(fixation_map_continuous(:));
    
    fprintf('    Computing AUC_Borji...\n');
    % AUC_Borji
    auc_borji = compute_auc_borji(saliency_map, fixation_map_binary);
    
    fprintf('    Computing AUC_Judd...\n');
    % AUC_Judd
    auc_judd = compute_auc_judd(saliency_map, fixation_map_binary);
    
    fprintf('    Computing AUC_shuffled...\n');
    % AUC_shuffled
    auc_shuffled = compute_auc_shuffled(saliency_map, fixation_map_binary);
    
    fprintf('    Computing CC...\n');
    % CC (Correlation Coefficient)
    cc = compute_cc(saliency_map, fixation_map_continuous);
    
    fprintf('    Computing EMD...\n');
    % EMD (Earth Mover's Distance)
    emd = compute_emd(saliency_map, fixation_map_continuous);
    
    fprintf('    Computing Info Gain...\n');
    % Info Gain
    info_gain = compute_info_gain(saliency_map, fixation_map_continuous);
    
    fprintf('    Computing KLdiv...\n');
    % KLdiv (Kullback-Leibler Divergence)
    kldiv = compute_kldiv(saliency_map, fixation_map_continuous);
    
    fprintf('    Computing NSS...\n');
    % NSS (Normalized Scanpath Saliency)
    nss = compute_nss(saliency_map, fixation_map_binary);
    
    fprintf('    Collecting all metrics...\n');
    % Collect all metrics
    metrics = [auc_borji, auc_judd, auc_shuffled, cc, emd, info_gain, kldiv, nss];
    
    fprintf('    Exiting compute_all_metrics function...\n');
end

% Function to compute AUC_Borji using alternative implementation
function auc = compute_auc_borji(saliency_map, fixation_map)
    fprintf('      Entering compute_auc_borji function...\n');
    
    % Default parameters
    Nsplits = 100;
    stepSize = 0.1;
    
    % Flatten the maps
    saliency_flat = saliency_map(:);
    fixation_flat = fixation_map(:);
    
    % Check if there are enough fixations
    fixation_indices = find(fixation_flat);
    if length(fixation_indices) <= 1
        auc = NaN;
        fprintf('      Not enough fixations, returning NaN\n');
        fprintf('      Exiting compute_auc_borji function...\n');
        return;
    end
    
    fprintf('      Found %d fixation points\n', length(fixation_indices));
    
    % Resize saliency map if necessary
    if size(saliency_map, 1) ~= size(fixation_map, 1) || size(saliency_map, 2) ~= size(fixation_map, 2)
        saliency_map = imresize(saliency_map, size(fixation_map));
        saliency_flat = saliency_map(:);
    end
    
    % Normalize saliency map
    saliency_map = (saliency_map - min(saliency_flat(:))) / (max(saliency_flat(:)) - min(saliency_flat(:)));
    saliency_flat = saliency_map(:);
    
    % Get saliency values at fixation locations
    Sth = saliency_flat(fixation_flat > 0);
    Nfixations = length(Sth);
    Npixels = length(saliency_flat);
    
    % Sample random locations
    fprintf('      Sampling random locations...\n');
    randfix = zeros(Nfixations, Nsplits);
    for s = 1:Nsplits
        random_indices = randi(Npixels, Nfixations, 1);
        randfix(:, s) = saliency_flat(random_indices);
    end
    
    % Compute AUC for each split
    fprintf('      Computing AUC for %d splits...\n', Nsplits);
    auc_values = zeros(1, Nsplits);
    for s = 1:Nsplits
        curfix = randfix(:, s);
        
        % Determine threshold range
        max_threshold = max([Sth; curfix]);
        allthreshes = fliplr(0:stepSize:double(max_threshold));
        
        % Initialize TP and FP arrays
        tp = zeros(length(allthreshes) + 2, 1);
        fp = zeros(length(allthreshes) + 2, 1);
        tp(1) = 0; tp(end) = 1;
        fp(1) = 0; fp(end) = 1;
        
        % Compute TP and FP rates
        for i = 1:length(allthreshes)
            thresh = allthreshes(i);
            tp(i+1) = sum(Sth >= thresh) / Nfixations;
            fp(i+1) = sum(curfix >= thresh) / Nfixations;
        end
        
        % Compute AUC for this split
        auc_values(s) = trapz(fp, tp);
    end
    
    % Average AUC across splits
    auc = mean(auc_values);
    
    fprintf('      AUC_Borji computed: %.4f\n', auc);
    fprintf('      Exiting compute_auc_borji function...\n');
end

% Function to compute AUC_Judd
function auc = compute_auc_judd(saliency_map, fixation_map)
    fprintf('      Entering compute_auc_judd function...\n');
    
    % Default parameters
    jitter = true;
    toPlot = false;
    
    % Flatten the maps
    saliency_flat = saliency_map(:);
    fixation_flat = fixation_map(:);
    
    % Check if there are fixations
    if ~any(fixation_flat)
        auc = NaN;
        fprintf('      No fixations found, returning NaN\n');
        fprintf('      Exiting compute_auc_judd function...\n');
        return;
    end
    
    % Resize saliency map if necessary
    if size(saliency_map, 1) ~= size(fixation_map, 1) || size(saliency_map, 2) ~= size(fixation_map, 2)
        saliency_map = imresize(saliency_map, size(fixation_map));
        saliency_flat = saliency_map(:);
    end
    
    % Add jitter if needed
    if jitter
        saliency_map = saliency_map + rand(size(saliency_map)) / 10000000;
        saliency_flat = saliency_map(:);
    end
    
    % Normalize saliency map
    saliency_map = (saliency_map - min(saliency_flat(:))) / (max(saliency_flat(:)) - min(saliency_flat(:)));
    saliency_flat = saliency_map(:);
    
    % Get saliency values at fixation locations
    Sth = saliency_flat(fixation_flat > 0);
    Nfixations = length(Sth);
    Npixels = length(saliency_flat);
    
    % Sort thresholds based on fixation locations
    allthreshes = sort(Sth, 'descend');
    
    % Initialize TP and FP arrays
    tp = zeros(Nfixations + 2, 1);
    fp = zeros(Nfixations + 2, 1);
    tp(1) = 0;
    tp(end) = 1;
    fp(1) = 0;
    fp(end) = 1;
    
    % Compute TP and FP rates
    for i = 1:Nfixations
        thresh = allthreshes(i);
        aboveth = sum(saliency_flat >= thresh);
        tp(i + 1) = i / Nfixations;
        fp(i + 1) = (aboveth - i) / (Npixels - Nfixations);
    end
    
    % Compute AUC
    auc = trapz(fp, tp);
    
    fprintf('      AUC_Judd computed: %.4f\n', auc);
    fprintf('      Exiting compute_auc_judd function...\n');
end

% Function to compute AUC_shuffled
% Function to compute AUC_shuffled
function auc = compute_auc_shuffled(saliency_map, fixation_map)
    fprintf('      Entering compute_auc_shuffled function...\n');
    
    % Default parameters
    Nsplits = 100;
    stepSize = 0.1;
    toPlot = false;
    
    % If there are no fixations to predict, return NaN
    if ~any(fixation_map(:))
        auc = NaN;
        fprintf('      No fixations found, returning NaN\n');
        fprintf('      Exiting compute_auc_shuffled function...\n');
        return;
    end
    
    % Make the saliencyMap the size of fixationMap
    if size(saliency_map, 1) ~= size(fixation_map, 1) || size(saliency_map, 2) ~= size(fixation_map, 2)
        saliency_map = imresize(saliency_map, size(fixation_map));
    end
    
    % Normalize saliency map
    saliency_map = (saliency_map - min(saliency_map(:))) / (max(saliency_map(:)) - min(saliency_map(:)));
    
    if sum(isnan(saliency_map(:))) == length(saliency_map(:))
        auc = NaN;
        fprintf('      NaN saliencyMap, returning NaN\n');
        fprintf('      Exiting compute_auc_shuffled function...\n');
        return;
    end
    
    S = saliency_map(:);
    F = fixation_map(:);
    
    Sth = S(F > 0); % sal map values at fixation locations
    Nfixations = length(Sth);
    
    % For each fixation, sample Nsplits values from the sal map at locations
    % specified by otherMap
    otherMap = get_other_fixations(fixation_map); % Implement this function to get fixations from other images
    
    ind = find(otherMap(:) > 0); % find fixation locations on other images
    
    Nfixations_oth = min(Nfixations, length(ind));
    randfix = NaN(Nfixations_oth, Nsplits);
    
    for i = 1:Nsplits
        randind = ind(randperm(length(ind))); % randomize choice of fixation locations
        randfix(:, i) = S(randind(1:Nfixations_oth)); % sal map values at random fixation locations of other random images
    end
    
    % Calculate AUC per random split (set of random locations)
    auc_values = NaN(1, Nsplits);
    for s = 1:Nsplits
        curfix = randfix(:, s);
        
        max_threshold = max([Sth; curfix]);
        allthreshes = fliplr(0:stepSize:double(max_threshold));
        
        tp = zeros(length(allthreshes) + 2, 1);
        fp = zeros(length(allthreshes) + 2, 1);
        tp(1) = 0;
        tp(end) = 1;
        fp(1) = 0;
        fp(end) = 1;
        
        for i = 1:length(allthreshes)
            thresh = allthreshes(i);
            tp(i+1) = sum(Sth >= thresh) / Nfixations;
            fp(i+1) = sum(curfix >= thresh) / Nfixations_oth;
        end
        
        auc_values(s) = trapz(fp, tp);
    end
    
    auc = mean(auc_values); % mean across random splits
    
    fprintf('      AUC_shuffled computed: %.4f\n', auc);
    fprintf('      Exiting compute_auc_shuffled function...\n');
end

% Function to get fixations from other images
function otherMap = get_other_fixations(current_fixation_map)
    % Implement this function to return fixations from other images
    % For demonstration, we'll return a random binary map of the same size
    otherMap = rand(size(current_fixation_map)) > 0.9;
end
% Function to compute CC (Correlation Coefficient)
function cc = compute_cc(saliency_map, fixation_map)
    fprintf('      Entering compute_cc function...\n');
    
    % Resize maps to the same size
    saliency_map = imresize(saliency_map, size(fixation_map));
    
    % Convert to double
    saliency_map = im2double(saliency_map);
    fixation_map = im2double(fixation_map);
    
    % Normalize both maps
    saliency_map = (saliency_map - mean(saliency_map(:))) / std(saliency_map(:));
    fixation_map = (fixation_map - mean(fixation_map(:))) / std(fixation_map(:));
    
    % Compute correlation coefficient
    cc = corr2(saliency_map, fixation_map);
    
    fprintf('      CC computed: %.4f\n', cc);
    fprintf('      Exiting compute_cc function...\n');
end


% Function to compute EMD (Earth Mover's Distance)
% Function to compute EMD (Earth Mover's Distance)
function emd = compute_emd(saliency_map, fixation_map)
    fprintf('      Entering compute_emd function...\n');
    
    % Check if Optimization Toolbox is available (i.e. linprog exists)
    if exist('linprog', 'file') ~= 2
        warning('Optimization Toolbox not available. Setting EMD to NaN.');
        emd = NaN;
        fprintf('      Exiting compute_emd function...\n');
        return;
    end

    % Default parameters
    downsize = 32;
    toPlot = false;
    
    % Reduce image size for efficiency
    im1 = imresize(fixation_map, 1/downsize);
    im2 = imresize(saliency_map, size(im1));
    
    % Normalize maps to sum to 1
    im1 = im1 / sum(im1(:));
    im2 = im2 / sum(im2(:));
    
    % Get map dimensions
    [R, C] = size(im1);
    
    % Create distance matrix
    D = zeros(R*C, R*C, 'double');
    j = 0;
    for c1 = 1:C
        for r1 = 1:R
            j = j + 1;
            i = 0;
            for c2 = 1:C
                for r2 = 1:R
                    i = i + 1;
                    D(i, j) = sqrt((r1 - r2)^2 + (c1 - c2)^2);
                end
            end
        end
    end
    
    % Convert maps to vectors
    P = double(im1(:));
    Q = double(im2(:));
    
    % Compute EMD using linear programming
    f = [P; Q];
    Aeq = [ones(1, length(P)), -ones(1, length(Q))];
    beq = 0;
    lb = zeros(length(P) + length(Q), 1);
    ub = [inf*ones(length(P), 1); Q];
    
    options = optimset('Display', 'none');
    flow = linprog(f, [], [], Aeq, beq, lb, ub, [], options);
    
    flow = flow(1:length(P));
    emd = sum(flow .* D(:)) / sum(P);
    
    if toPlot
        figure(1)
        subplot(221); imshow(fixation_map);
        subplot(222); imshow(saliency_map);
        subplot(223); imshow(im1, []); title(['EMD: ', num2str(emd)])
        subplot(224); imshow(im2, []);
        
        figure(2)
        subplot(131); imshow([im1; im2], []); title('Resized Maps');
        subplot(132); imhist(im1); title('Fixation Map Histogram');
        subplot(133); imhist(im2); title('Saliency Map Histogram');
    end
    
    fprintf('      EMD computed: %.4f\n', emd);
    fprintf('      Exiting compute_emd function...\n');
end


% Function to compute Info Gain
function info_gain = compute_info_gain(saliency_map, fixation_map)
    fprintf('      Entering compute_info_gain function...\n');
    
    % Resize maps to the same size
    saliency_map = imresize(saliency_map, size(fixation_map));
    
    % Normalize maps
    saliency_map = (saliency_map - min(saliency_map(:))) / (max(saliency_map(:)) - min(saliency_map(:)));
    fixation_map = (fixation_map - min(fixation_map(:))) / (max(fixation_map(:)) - min(fixation_map(:)));
    
    % Convert to distributions
    saliency_map = saliency_map / sum(saliency_map(:));
    fixation_map = fixation_map / sum(fixation_map(:));
    
    % Get fixation locations
    locs = logical(fixation_map(:));
    
    % Compute information gain
    info_gain = mean(log2(eps + saliency_map(locs)) - log2(eps + fixation_map(locs)));
    
    fprintf('      Info Gain computed: %.4f\n', info_gain);
    fprintf('      Exiting compute_info_gain function...\n');
end

% Function to compute KLdiv (Kullback-Leibler Divergence)
function kldiv = compute_kldiv(saliency_map, fixation_map)
    fprintf('      Entering compute_kldiv function...\n');
    
    % Resize maps to the same size
    saliency_map = imresize(saliency_map, size(fixation_map));
    
    % Normalize maps to sum to 1
    if any(saliency_map(:))
        saliency_map = saliency_map / sum(saliency_map(:));
    end
    
    if any(fixation_map(:))
        fixation_map = fixation_map / sum(fixation_map(:));
    end
    
    % Compute KL divergence
    kldiv = sum(fixation_map(:) .* log2(eps + fixation_map(:) ./ (saliency_map(:) + eps)));
    
    fprintf('      KLdiv computed: %.4f\n', kldiv);
    fprintf('      Exiting compute_kldiv function...\n');
end

% Function to compute NSS (Normalized Scanpath Saliency)
function nss = compute_nss(saliency_map, fixation_map)
    fprintf('      Entering compute_nss function...\n');
    
    % Resize maps to the same size
    saliency_map = imresize(saliency_map, size(fixation_map));
    
    % Normalize saliency map
    saliency_map = (saliency_map - mean(saliency_map(:))) / std(saliency_map(:));
    
    % Get fixation locations
    locs = logical(fixation_map(:));
    
    % Compute NSS
    nss = mean(saliency_map(locs));
    
    fprintf('      NSS computed: %.4f\n', nss);
    fprintf('      Exiting compute_nss function...\n');
end


% Function to apply Antonio's Gaussian filter
function filtered = antonioGaussian(img, fc)
    fprintf('      Entering antonioGaussian function...\n');
    
    % Convert image to double
    img = double(img);
    
    % Get image dimensions
    [rows, cols] = size(img);
    
    % Create frequency domain coordinates
    [X, Y] = meshgrid(1:cols, 1:rows);
    X = X - floor(cols/2) - 1;
    Y = Y - floor(rows/2) - 1;
    
    % Create Gaussian filter
    radius = sqrt(X.^2 + Y.^2);
    filter = exp(-(radius.^2) / (2 * (fc / 2.5)^2));
    
    % Apply filter in frequency domain
    img_fft = fft2(img);
    filtered_fft = img_fft .* filter;
    filtered = real(ifft2(filtered_fft));
    
    % Normalize result
    filtered = (filtered - min(filtered(:))) / (max(filtered(:)) - min(filtered(:)));
    
    fprintf('      Exiting antonioGaussian function...\n');
end