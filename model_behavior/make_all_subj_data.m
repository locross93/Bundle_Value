% load data for each subject
clear;

data_dir = '/Users/locro/Documents/Bundle_Value/model_behavior/data/';

sub_list = {'101','102','103','104','105','106','107','108','109','110','111','112','113','114'};
%sub_list = {'101','102','103','104','105','106'};
num_subjs = length(sub_list);
data = cell(num_subjs, 1);
for sub_num=1:num_subjs
    subID = sub_list{sub_num};
    temp_file = [data_dir,'choice_behavior_sub',subID];
    subj_df = readtable(temp_file);
    subj_struct = table2struct(subj_df, "ToScalar",true);
    data{sub_num} = subj_struct;
end

% save
save([data_dir,'all_subj_data.mat'], 'data')