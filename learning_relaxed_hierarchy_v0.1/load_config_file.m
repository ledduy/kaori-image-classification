function config = load_config_file(filename)

% set default values
config.confusion_matrix_fn = '';
config.C = 1;
config.rho = 0.6;
config.num_samples = 30;
config.hierarchy_level = 10;
config.init_method = 'bottom_up';
config.C_one_vs_all = 1;
config.max_num_iter = 2;
config.msg_level = 'verbose';
config.kernel_type = 'linear';
config.num_proc = 1;
config.B = 10;
config.alpha = 0.6;

% open file
fid = fopen(filename, 'rt');
lines = textscan(fid, '%s');
lines = lines{1};
fclose(fid);

% set configurations
for i = 1 : length(lines)
  if strcmp(lines{i}(1), '#')
     switch lower(lines{i}(2:end))
         case 'confusion_matrix_fn'
             config.confusion_matrix_fn = lines{i+1};
         case 'c'
             config.C = str2double(lines{i+1});
         case 'b'
             config.B = str2double(lines{i+1});
         case 'rho'
             config.rho = str2double(lines{i+1});
         case 'num_samples'
             config.num_samples = str2double(lines{i+1});
         case 'hierarchy_level'
             config.hierarchy_level = str2double(lines{i+1});
         case 'init_method'
             config.init_method = lines{i+1};
         case 'c_one_vs_all'
             config.C_one_vs_all = str2double(lines{i+1});
         case 'max_num_iter'
             config.max_num_iter = str2double(lines{i+1});
         case 'msg_level'
             config.msg_level = lines{i+1};
         case 'kernel_type'
             config.kernel_type = lines{i+1};
         case 'num_proc'
             config.num_proc = str2double(lines{i+1});
         case 'alpha'
             config.alpha = str2double(lines{i+1});             
     end
  end
end
