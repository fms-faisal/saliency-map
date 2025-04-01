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
options.size = 512;                     % Resize image
options.quantile = 1/10;                % Neighborhood similarity
options.centerBias = 1;                 % Enable center bias
options.modeltype = 'CovariancesOnly';  % Use only covariance features

%% Initialize metric storage
num_images = length(image_files);
num_metrics = 8;  % 8 evaluation metrics

td_metrics = zeros(num_images, num_metrics);
asd_metrics = zeros(num_images, num_metrics);

%% Generate and save saliency maps
for i = 1:num_images
    img_name = image_files(i).name;
    img_path = fullfile(image_folder, img_name);

    try
        % Compute saliency map
        sal_map = saliencymap(img_path, options);
        
        % Normalize saliency map to [0, 1]
        sal_map = (sal_map - min(sal_map(:))) / (max(sal_map(:)) - min(sal_map(:)));
        
        % Convert to uint8 for proper image saving
        sal_map_uint8 = im2uint8(sal_map);
        
        % Save saliency map
        output_path = fullfile(output_folder, ['Saliency_' img_name]);
        imwrite(sal_map_uint8, output_path);
        
    catch ME
        warning('Error processing %s: %s', img_name, ME.message);
        continue;
    end
end

disp('Saliency maps generated.');

% %% Evaluation: Compare with TD and ASD Fixation Maps
% for i = 1:num_images
%     img_name = image_files(i).name;
%     [~, base_name, ~] = fileparts(img_name);  % Get base name (e.g., '1' from '1.png')
%     pred_path = fullfile(output_folder, ['Saliency_' img_name]);
% 
%     % Construct fixation map filenames with "_s.png"
%     td_fix_name = [base_name '_s.png'];
%     asd_fix_name = [base_name '_s.png'];
%     td_fix_path = fullfile(td_fixation_folder, td_fix_name);
%     asd_fix_path = fullfile(asd_fixation_folder, asd_fix_name);
% 
%     % Check file existence
%     if ~all(exist([pred_path, td_fix_path, asd_fix_path], 'file'))
%         warning('Missing files for %s', img_name);
%         continue;
%     end
% 
%     try
%         % Read images with proper format handling
%         pred_map = im2double(imread(pred_path));  % Convert to double [0,1]
%         td_fix_map = im2double(imread(td_fix_path));
%         asd_fix_map = im2double(imread(asd_fix_path));
% 
%         % Ensure all maps are the same size
%         [h, w] = size(pred_map);
%         td_fix_map = imresize(td_fix_map, [h w]);
%         asd_fix_map = imresize(asd_fix_map, [h w]);
% 
%         % Compute evaluation metrics
%         td_metrics(i, :) = computeAllMetrics(pred_map, td_fix_map);
%         asd_metrics(i, :) = computeAllMetrics(pred_map, asd_fix_map);
% 
%     catch ME
%         warning('Evaluation error for %s: %s', img_name, ME.message);
%         continue;
%     end
% end
% 
% % Compute mean performance
% if ~isempty(td_metrics)
%     avg_td_metrics = mean(td_metrics, 1);
% else
%     avg_td_metrics = NaN(1, num_metrics);
% end
% 
% if ~isempty(asd_metrics)
%     avg_asd_metrics = mean(asd_metrics, 1);
% else
%     avg_asd_metrics = NaN(1, num_metrics);
% end
% 
% %% Display Results
% fprintf('TD Evaluation Metrics (Saliency Model - CovariancesOnly):\n');
% fprintf('AUC_Borji: %.4f, AUC_Judd: %.4f, AUC_shuffled: %.4f, CC: %.4f, EMD: %.4f, InfoGain: %.4f, KLdiv: %.4f, NSS: %.4f\n\n', ...
%     avg_td_metrics(1), avg_td_metrics(2), avg_td_metrics(3), avg_td_metrics(4), avg_td_metrics(5), avg_td_metrics(6), avg_td_metrics(7), avg_td_metrics(8));
% 
% fprintf('ASD Evaluation Metrics (Saliency Model - CovariancesOnly):\n');
% fprintf('AUC_Borji: %.4f, AUC_Judd: %.4f, AUC_shuffled: %.4f, CC: %.4f, EMD: %.4f, InfoGain: %.4f, KLdiv: %.4f, NSS: %.4f\n', ...
%     avg_asd_metrics(1), avg_asd_metrics(2), avg_asd_metrics(3), avg_asd_metrics(4), avg_asd_metrics(5), avg_asd_metrics(6), avg_asd_metrics(7), avg_asd_metrics(8));
% 
% %% Helper function to compute all metrics
% function metrics = computeAllMetrics(pred_map, fix_map)
%     metrics(1) = AUC_Borji(pred_map, fix_map);
%     metrics(2) = AUC_Judd(pred_map, fix_map);
%     metrics(3) = AUC_shuffled(pred_map, fix_map);
%     metrics(4) = CC(pred_map, fix_map);
%     metrics(5) = EMD(pred_map, fix_map);
%     metrics(6) = InfoGain(pred_map, fix_map);
%     metrics(7) = KLdiv(pred_map, fix_map);
%     metrics(8) = NSS(pred_map, fix_map);
% end