%% Clear workspace
clear all;
clc;

%% Define folders
image_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\Images';
td_fixation_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\TD_FixMaps';
asd_fixation_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency4ASD\ASD_FixMaps';
output_folder = 'E:\Master''s Courses\CSE583 SFR1 Spring 2025\Assignment_03\Saliency4ASD\Saliency_Maps';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get list of images
image_files = dir(fullfile(image_folder, '*.png'));

%% Options for Saliency Estimation
options.size = 512;                      % Target size for images
options.quantile = 1/10;                % Quantile for thresholding
options.centerBias = 1;                 % Apply center bias correction
options.modeltype = 'CovariancesOnly';  % Model type for saliency estimation

%% Initialize metric storage
num_images = length(image_files);
num_metrics = 8;  % Number of evaluation metrics

td_metrics = zeros(num_images, num_metrics);
asd_metrics = zeros(num_images, num_metrics);

%% Generate and save saliency maps
% Implement saliencymap function for covariance-based saliency
function sal_map = saliencymap(img_path, options)
    % Read and preprocess image
    img = imread(img_path);
    img = im2double(img);
    
    % Resize image to target size
    img = imresize(img, [options.size options.size]);
    
    % Convert to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Compute covariance matrix
    [rows, cols] = size(img);
    img_vec = reshape(img, [], 1);
    mean_val = mean(img_vec);
    centered_img = img_vec - mean_val;
    cov_mat = (centered_img * centered_img') / (numel(img) - 1);
    
    % Compute saliency map based on covariance
    sal_map = zeros(size(img));
    for i = 1:rows
        for j = 1:cols
            % Compute local covariance around each pixel
            local_patch = img(max(1, i-5):min(rows, i+5), max(1, j-5):min(cols, j+5));
            local_vec = reshape(local_patch, [], 1);
            local_mean = mean(local_vec);
            local_centered = local_vec - local_mean;
            local_cov = (local_centered * local_centered') / (numel(local_patch) - 1);
            
            % Compute distance between global and local covariance
            sal_map(i,j) = norm(cov_mat - local_cov);
        end
    end
    
    % Normalize saliency map
    sal_map = (sal_map - min(sal_map(:))) / (max(sal_map(:)) - min(sal_map(:)) + eps);
end

% Generate saliency maps
for i = 1:num_images
    img_name = image_files(i).name;
    img_path = fullfile(image_folder, img_name);
    
    try
        sal_map = saliencymap(img_path, options);
        sal_map_uint8 = im2uint8(sal_map);
        output_path = fullfile(output_folder, ['Saliency_' img_name]);
        imwrite(sal_map_uint8, output_path);
    catch ME
        warning('Error processing %s: %s', img_name, ME.message);
        continue;
    end
end
disp('Saliency maps generated.');

%% Evaluation: Compare with TD and ASD Fixation Maps
for i = 1:num_images
    img_name = image_files(i).name;
    [~, base_name, ~] = fileparts(img_name);
    pred_path = fullfile(output_folder, ['Saliency_' img_name]);

    % Construct fixation map filenames (adjust suffixes as needed)
    td_fix_name = [base_name '_td_fix.png'];  % Adjust suffix for TD
    asd_fix_name = [base_name '_asd_fix.png']; % Adjust suffix for ASD
    td_fix_path = fullfile(td_fixation_folder, td_fix_name);
    asd_fix_path = fullfile(asd_fixation_folder, asd_fix_name);

    % Check if all files exist
    if ~exist(pred_path, 'file')
        warning('Missing saliency map for %s', img_name);
        continue;
    end
    if ~exist(td_fix_path, 'file')
        warning('Missing TD fixation map for %s', img_name);
        continue;
    end
    if ~exist(asd_fix_path, 'file')
        warning('Missing ASD fixation map for %s', img_name);
        continue;
    end

    try
        % Read and convert to double
        pred_map = im2double(imread(pred_path));
        
        % Read fixation maps and convert to grayscale if necessary
        td_fix_map = im2double(imread(td_fix_path));
        if size(td_fix_map, 3) == 3
            td_fix_map = rgb2gray(td_fix_map);
        end
        
        asd_fix_map = im2double(imread(asd_fix_path));
        if size(asd_fix_map, 3) == 3
            asd_fix_map = rgb2gray(asd_fix_map);
        end
    
        % Resize fixation maps to match saliency map size
        [h, w] = size(pred_map);
        td_fix_map = imresize(td_fix_map, [h w], 'nearest');
        asd_fix_map = imresize(asd_fix_map, [h w], 'nearest');
    
        % Ensure fixation maps are binary
        td_fix_map = td_fix_map > 0;
        asd_fix_map = asd_fix_map > 0;
    
        % Compute metrics
        td_metrics(i, :) = computeAllMetrics(pred_map, td_fix_map);
        asd_metrics(i, :) = computeAllMetrics(pred_map, asd_fix_map);
        
    catch ME
        warning('Evaluation error for %s: %s', img_name, ME.message);
        continue;
    end
end

% Compute mean performance
if ~isempty(td_metrics)
    avg_td_metrics = mean(td_metrics, 1);
else
    avg_td_metrics = NaN(1, num_metrics);
end

if ~isempty(asd_metrics)
    avg_asd_metrics = mean(asd_metrics, 1);
else
    avg_asd_metrics = NaN(1, num_metrics);
end

%% Display Results
fprintf('TD Evaluation Metrics:\n');
fprintf('AUC_Borji: %.4f, AUC_Judd: %.4f, AUC_shuffled: %.4f, CC: %.4f, EMD: %.4f, InfoGain: %.4f, KLdiv: %.4f, NSS: %.4f\n\n', ...
    avg_td_metrics(1), avg_td_metrics(2), avg_td_metrics(3), avg_td_metrics(4), avg_td_metrics(5), avg_td_metrics(6), avg_td_metrics(7), avg_td_metrics(8));

fprintf('ASD Evaluation Metrics:\n');
fprintf('AUC_Borji: %.4f, AUC_Judd: %.4f, AUC_shuffled: %.4f, CC: %.4f, EMD: %.4f, InfoGain: %.4f, KLdiv: %.4f, NSS: %.4f\n', ...
    avg_asd_metrics(1), avg_asd_metrics(2), avg_asd_metrics(3), avg_asd_metrics(4), avg_asd_metrics(5), avg_asd_metrics(6), avg_asd_metrics(7), avg_asd_metrics(8));

%% Helper function to compute all metrics
function metrics = computeAllMetrics(pred_map, fix_map)
    metrics(1) = AUC_Borji(pred_map, fix_map);
    metrics(2) = AUC_Judd(pred_map, fix_map);
    metrics(3) = AUC_shuffled(pred_map, fix_map);
    metrics(4) = CC(pred_map, fix_map);
    metrics(5) = EMD(pred_map, fix_map);
    metrics(6) = InfoGain(pred_map, fix_map);
    metrics(7) = KLdiv(pred_map, fix_map);
    metrics(8) = NSS(pred_map, fix_map);
end

%% Implementation of evaluation metrics
function auc = AUC_Borji(saliency_map, fixation_map)
    fixation_map = fixation_map > 0;
    [rows, cols] = size(saliency_map);
    num_fixations = sum(fixation_map(:));
    
    if num_fixations == 0
        auc = NaN;
        return;
    end
    
    thresholds = unique(saliency_map);
    thresholds = sort(thresholds, 'descend');
    
    num_splits = 100;
    auc = 0;
    
    for split = 1:num_splits
        non_fixation_indices = find(~fixation_map);
        max_samples = min(num_fixations, length(non_fixation_indices));
        
        if max_samples == 0
            fp = zeros(size(thresholds));
        else
            rand_indices = randperm(length(non_fixation_indices), max_samples);
            rand_non_fixation = non_fixation_indices(rand_indices);
            rand_fixation_map = false(rows, cols);
            rand_fixation_map(rand_non_fixation) = true;
            
            fp = sum((saliency_map >= thresholds) & rand_fixation_map, 'all');
        end
        
        tp = sum((saliency_map >= thresholds) & fixation_map, 'all');
        tpr = tp / num_fixations;
        fpr = fp / max_samples;
        auc = auc + trapz(fpr, tpr);
    end
    auc = auc / num_splits;
end

function auc = AUC_Judd(saliency_map, fixation_map)
    fixation_map = fixation_map > 0;
    [rows, cols] = size(saliency_map);
    num_fixations = sum(fixation_map(:));
    
    if num_fixations == 0
        auc = NaN;
        return;
    end
    
    thresholds = unique(saliency_map);
    thresholds = sort(thresholds, 'descend');
    
    tp = sum((saliency_map >= thresholds) & fixation_map, 'all');
    fp = sum((saliency_map >= thresholds) & ~fixation_map, 'all');
    
    tpr = tp / num_fixations;
    fpr = fp / (rows*cols - num_fixations);
    auc = trapz(fpr, tpr);
end

function auc = AUC_shuffled(saliency_map, fixation_map)
    fixation_map = fixation_map > 0;
    [rows, cols] = size(saliency_map);
    num_fixations = sum(fixation_map(:));
    
    if num_fixations == 0
        auc = NaN;
        return;
    end
    
    thresholds = unique(saliency_map);
    thresholds = sort(thresholds, 'descend');
    
    num_splits = 100;
    auc = 0;
    
    for split = 1:num_splits
        fixation_indices = find(fixation_map);
        non_fixation_indices = find(~fixation_map);
        rand_indices = [fixation_indices; non_fixation_indices(randperm(length(non_fixation_indices), num_fixations))];
        
        shuffled_fixation_map = false(rows, cols);
        shuffled_fixation_map(rand_indices(1:num_fixations)) = true;
        
        tp = sum((saliency_map >= thresholds) & shuffled_fixation_map, 'all');
        fp = sum((saliency_map >= thresholds) & ~shuffled_fixation_map, 'all');
        
        tpr = tp / num_fixations;
        fpr = fp / (rows*cols - num_fixations);
        auc = auc + trapz(fpr, tpr);
    end
    auc = auc / num_splits;
end

function cc = CC(saliency_map, fixation_map)
    saliency_map = (saliency_map - mean(saliency_map(:))) / (std(saliency_map(:)) + eps);
    fixation_map = (fixation_map - mean(fixation_map(:))) / (std(fixation_map(:)) + eps);
    cc = mean(saliency_map(:) .* fixation_map(:));
end

function emd = EMD(saliency_map, fixation_map)
    saliency_map = saliency_map(:) / sum(saliency_map(:));
    fixation_map = fixation_map(:) / sum(fixation_map(:));
    
    [Y, X] = meshgrid(1:size(saliency_map,2), 1:size(saliency_map,1));
    coordinates = [X(:) Y(:)];
    cost_matrix = pdist2(coordinates, coordinates, 'euclidean');
    
    emd = emd2(saliency_map, fixation_map, cost_matrix);
end

function ig = InfoGain(saliency_map, fixation_map)
    fixation_map = fixation_map > 0;
    num_fixations = sum(fixation_map(:));
    
    if num_fixations == 0
        ig = NaN;
        return;
    end
    
    p_fix = num_fixations / numel(fixation_map);
    entropy_fix = -p_fix*log2(p_fix) - (1-p_fix)*log2(1-p_fix);
    
    saliency_map = (saliency_map - min(saliency_map(:))) / (max(saliency_map(:)) - min(saliency_map(:)) + eps);
    thresholds = unique(saliency_map);
    thresholds = sort(thresholds, 'descend');
    
    max_ig = -inf;
    for i = 1:length(thresholds)
        binary_map = saliency_map >= thresholds(i);
        p_sal = sum(binary_map(:)) / numel(binary_map);
        
        p_sal_fix = sum(binary_map(fixation_map)) / numel(binary_map);
        p_not_sal_fix = sum(~binary_map(fixation_map)) / numel(binary_map);
        p_sal_not_fix = sum(binary_map(~fixation_map)) / numel(binary_map);
        p_not_sal_not_fix = sum(~binary_map(~fixation_map)) / numel(binary_map);
        
        conditional_entropy = 0;
        if p_sal > 0
            conditional_entropy = conditional_entropy - p_sal_fix * log2(p_sal_fix / p_sal);
        end
        if (1-p_sal) > 0
            conditional_entropy = conditional_entropy - p_not_sal_fix * log2(p_not_sal_fix / (1-p_sal));
        end
        ig_current = entropy_fix - conditional_entropy;
        
        if ig_current > max_ig
            max_ig = ig_current;
        end
    end
    ig = max_ig;
end

function kl = KLdiv(saliency_map, fixation_map)
    saliency_map = saliency_map(:) + eps;
    saliency_map = saliency_map / sum(saliency_map);
    
    fixation_map = fixation_map(:) + eps;
    fixation_map = fixation_map / sum(fixation_map);
    
    kl = sum(fixation_map .* log(fixation_map ./ saliency_map));
end

function nss = NSS(saliency_map, fixation_map)
    fixation_map = fixation_map > 0;
    num_fixations = sum(fixation_map(:));
    
    if num_fixations == 0
        nss = NaN;
        return;
    end
    
    saliency_map = (saliency_map - mean(saliency_map(:))) / (std(saliency_map(:)) + eps);
    nss = mean(saliency_map(fixation_map));
end

function emd = emd2(p, q, C)
    p = p(:);
    q = q(:);
    C = C(:,:);
    
    n = length(p);
    f = zeros(2*n, 1);
    Aeq = [speye(n), -speye(n)];
    beq = p - q;
    
    options = optimset('Algorithm', 'interior-point', 'Display', 'none');
    lb = zeros(2*n, 1);
    [x, ~] = linprog(f, [], [], Aeq, beq, lb, [], [], [], options);
    emd = x' * C(:); 
end