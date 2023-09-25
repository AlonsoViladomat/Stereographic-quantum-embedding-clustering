close all;
clear; 
clc;


filename = 'data_2_7.txt';
% filename = 'data_6_6.txt';
% filename = 'data_8_6.txt';
%  filename = 'data_10_7.txt';

f = fopen(filename);
data = textscan(f,'%s');
fclose(f);
variable = str2double(data{1}(1:1:end));
resultsdata = zeros(12000,5);

for i = 1:12000
    resultsdata(i,1) = variable(9*i-8);
    resultsdata(i,2) = variable(9*i-7);
    resultsdata(i,3) = variable(9*i-6);
    resultsdata(i,4) = variable(9*i-5);
    resultsdata(i,5) = variable(9*i-4);
    resultsdata(i,6) = variable(9*i-3);
    resultsdata(i,7) = variable(9*i-2);
    resultsdata(i,8) = variable(9*i-1);
    resultsdata(i,9) = variable(9*i);
end

% Data Metrics
% # radius 
% # no. of points 
% # mean of (training accuracy - testing accuracy)
% # mean training accuracy 
% # std dev of training accuracy 
% # mean testing accuracy 
% # std dev of testing accuracy 
% # mean no. of training iter 
% # std dev of no. of training iter
% # mean no. of testing iter 
% # std dev of no. of testing iter
% # mean execution time for training
% # std dev of execution time for training
% # mean execution time for testing
% # std dev of execution time for testing

datametric = zeros(120,15);

for j = 1:120
    
   index1 = 100*(j-1) + 1;
   index2 = 100*j;
   
   diff_acc = resultsdata(index1:index2, 3);
   training_acc = resultsdata(index1:index2, 4);
   testing_acc = resultsdata(index1:index2, 5);
   training_num_iter = resultsdata(index1:index2, 6);
   testing_num_iter = resultsdata(index1:index2, 7);
   training_exec_time = resultsdata(index1:index2, 8);
   testing_exec_time = resultsdata(index1:index2, 9);
   
   datametric(j,1) = resultsdata(index1, 1);
   datametric(j,2) = resultsdata(index1, 2);
   datametric(j,3) = mean(diff_acc);
   datametric(j,4) = mean(training_acc);
   datametric(j,5) = std(training_acc);
   datametric(j,6) = mean(testing_acc);
   datametric(j,7) = std(testing_acc);
   datametric(j,8) = mean(training_num_iter);
   datametric(j,9) = std(training_num_iter);
   datametric(j,10) = mean(testing_num_iter);
   datametric(j,11) = std(testing_num_iter);
   datametric(j,12) = mean(training_exec_time);
   datametric(j,13) = std(training_exec_time);
   datametric(j,14) = mean(testing_exec_time);
   datametric(j,15) = std(testing_exec_time);
   
end

datametric_ahead = datametric; 

datametric(:,3) = round(datametric(:,3),3);
datametric(:,4) = round(datametric(:,4),3);
datametric(:,5) = round(datametric(:,5),3);
datametric(:,6) = round(datametric(:,6),3);
datametric(:,7) = round(datametric(:,7),3);
datametric(:,8) = round(datametric(:,8),3);
datametric(:,9) = round(datametric(:,9),3);
datametric(:,10) = round(datametric(:,10),3);
datametric(:,11) = round(datametric(:,11),3);
datametric(:,12) = round(1000*datametric(:,12),3); %storing time in milliseconds
datametric(:,13) = round(1000*datametric(:,13),3); %storing time in milliseconds
datametric(:,14) = round(1000*datametric(:,14),3); %storing time in milliseconds
datametric(:,15) = round(1000*datametric(:,15),3); %storing time in milliseconds


%% storing complete processed result 
filenamestore = 'complete_processed_result_stereographic_overfitting_test_2_7.txt'; 
% filenamestore = 'complete_processed_result_stereographic_overfitting_test_6_6.txt'; 
% filenamestore = 'complete_processed_result_stereographic_overfitting_test_8_6.txt'; 
%  filenamestore = 'complete_processed_result_stereographic_overfitting_test_10_7.txt'; 

writematrix(datametric,filenamestore,'Delimiter','\t');  
type complete_processed_result_stereographic_overfitting_test_2_7.txt;
% type complete_processed_result_stereographic_overfitting_test_6_6.txt;
% type complete_processed_result_stereographic_overfitting_test_8_6.txt;
%  type complete_processed_result_stereographic_overfitting_test_10_7.txt;

xlfilenamestore = 'complete_processed_result_stereographic_overfitting_test_2_7.xlsx'; 
% xlfilenamestore = 'complete_processed_result_stereographic_overfitting_test_6_6.xlsx'; 
% xlfilenamestore = 'complete_processed_result_stereographic_overfitting_test_8_6.xlsx'; 
%  xlfilenamestore = 'complete_processed_result_stereographic_overfitting_test_10_7.xlsx'; 

xlswrite(xlfilenamestore,datametric);



%% Further processing of results

datametric_acc = datametric_ahead;
datametric_time = datametric_ahead;

for k = 1:8
    
   index3 = 15*(k-1) + 1;
   index4 = 15*k;
   A = datametric_acc(index3:index4, :);
   A = sortrows(A,6,'descend');
   datametric_acc(index3:index4, :) = A;
   
end

for l = 1:8
    
   index3 = 15*(l-1) + 1;
   index4 = 15*l;
   B = datametric_time(index3:index4, :);
   B = sortrows(B,12,'ascend');
   datametric_time(index3:index4, :) = B;
   
end
% 
% datametric_acc(:,3) = round(datametric_acc(:,3),3);
% datametric_acc(:,4) = round(datametric_acc(:,4),3);
% datametric_acc(:,7) = 1000*datametric_acc(:,7); %storing time in milliseconds
% datametric_acc(:,8) = 1000*datametric_acc(:,8); %storing time in milliseconds
% datametric_acc(:,7) = round(datametric_acc(:,7),3);
% 
% datametric_time(:,3) = round(datametric_time(:,3),3);
% datametric_time(:,4) = round(datametric_time(:,4),3);
% datametric_time(:,7) = 1000*datametric_time(:,7); %storing time in milliseconds
% datametric_time(:,8) = 1000*datametric_time(:,8); %storing time in milliseconds
% datametric_time(:,7) = round(datametric_time(:,7),3);


datametric_acc(:,3) = round(datametric_acc(:,3),3);
datametric_acc(:,4) = round(datametric_acc(:,4),3);
datametric_acc(:,5) = round(datametric_acc(:,5),3);
datametric_acc(:,6) = round(datametric_acc(:,6),3);
datametric_acc(:,7) = round(datametric_acc(:,7),3);
datametric_acc(:,8) = round(datametric_acc(:,8),3);
datametric_acc(:,9) = round(datametric_acc(:,9),3);
datametric_acc(:,10) = round(datametric_acc(:,10),3);
datametric_acc(:,11) = round(datametric_acc(:,11),3);
datametric_acc(:,12) = round(1000*datametric_acc(:,12),3); %storing time in milliseconds
datametric_acc(:,13) = round(1000*datametric_acc(:,13),3); %storing time in milliseconds
datametric_acc(:,14) = round(1000*datametric_acc(:,14),3); %storing time in milliseconds
datametric_acc(:,15) = round(1000*datametric_acc(:,15),3); %storing time in milliseconds


datametric_time(:,3) = round(datametric_time(:,3),3);
datametric_time(:,4) = round(datametric_time(:,4),3);
datametric_time(:,5) = round(datametric_time(:,5),3);
datametric_time(:,6) = round(datametric_time(:,6),3);
datametric_time(:,7) = round(datametric_time(:,7),3);
datametric_time(:,8) = round(datametric_time(:,8),3);
datametric_time(:,9) = round(datametric_time(:,9),3);
datametric_time(:,10) = round(datametric_time(:,10),3);
datametric_time(:,11) = round(datametric_time(:,11),3);
datametric_time(:,12) = round(1000*datametric_time(:,12),3); %storing time in milliseconds
datametric_time(:,13) = round(1000*datametric_time(:,13),3); %storing time in milliseconds
datametric_time(:,14) = round(1000*datametric_time(:,14),3); %storing time in milliseconds
datametric_time(:,15) = round(1000*datametric_time(:,15),3); %storing time in milliseconds


winner_acc = zeros(8,15);
winner_time = zeros(8,15);

for m = 1:8
    index2 = 15*(m-1)+1;
    winner_acc(m,:) = datametric_acc(index2,:);
    winner_time(m,:) = datametric_time(index2,:); 
end

%% storing processed result sorted by accuracy
filenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_2_7.txt'; 
% filenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_6_6.txt'; 
% filenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_8_6.txt'; 
%  filenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_10_7.txt'; 

writematrix(winner_acc,filenamestore,'Delimiter','\t');  
type accuracy_processed_result_timing_sim_stereo_classical_2_7.txt;
% type accuracy_processed_result_timing_sim_stereo_classical_6_6.txt;
% type accuracy_processed_result_timing_sim_stereo_classical_8_6.txt;
%  type accuracy_processed_result_timing_sim_stereo_classical_10_7.txt;

xlfilenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_2_7.xlsx'; 
% xlfilenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_6_6.xlsx'; 
% xlfilenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_8_6.xlsx'; 
%  xlfilenamestore = 'accuracy_processed_result_timing_sim_stereo_classical_10_7.xlsx'; 

xlswrite(xlfilenamestore,winner_acc);
 
%% storing processed result sorted by time 

filenamestore = 'time_processed_result_timing_sim_stereo_classical_2_7.txt'; 
% filenamestore = 'time_processed_result_timing_sim_stereo_classical_6_6.txt'; 
% filenamestore = 'time_processed_result_timing_sim_stereo_classical_8_6.txt'; 
%  filenamestore = 'time_processed_result_timing_sim_stereo_classical_10_7.txt'; 

writematrix(winner_time,filenamestore,'Delimiter','\t');  
type time_processed_result_timing_sim_stereo_classical_2_7.txt;
% type time_processed_result_timing_sim_stereo_classical_6_6.txt;
% type time_processed_result_timing_sim_stereo_classical_8_6.txt;
%  type time_processed_result_timing_sim_stereo_classical_10_7.txt;

xlfilenamestore = 'time_processed_result_timing_sim_stereo_classical_2_7.xlsx'; 
% xlfilenamestore = 'time_processed_result_timing_sim_stereo_classical_6_6.xlsx'; 
% xlfilenamestore = 'time_processed_result_timing_sim_stereo_classical_8_6.xlsx'; 
%  xlfilenamestore = 'time_processed_result_timing_sim_stereo_classical_10_7.xlsx'; 

xlswrite(xlfilenamestore,winner_time); 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

% 
% Radii = datametric(:,1);
% No_of_Points =  datametric(:,2);
% Average_Accuracy = datametric(:,3);
% Standard_Deviation_Average_Accuracy =  datametric(:,4);
% Average_Balanced_Accuracy = datametric(:,5);
% Standard_Deviation_Balanced_Accuracy =  datametric(:,6);
% Average_No_of_iterations = datametric(:,7);
% Standard_Deviation_No_of_iterations =  datametric(:,8);
% 
% T = table(Radii , No_of_Points , Average_Accuracy , Standard_Deviation_Average_Accuracy , Average_Balanced_Accuracy , Standard_Deviation_Balanced_Accuracy , Average_No_of_iterations , Standard_Deviation_No_of_iterations );
% 
% filenamestore = 'processed_result_stereo_classical_2_7.txt'; 
% % filenamestore = 'processed_result_stereo_classical_6_6.txt'; 
% % filenamestore = 'processed_result_stereo_classical_8_6.txt'; 
% % filenamestore = 'processed_result_stereo_classical_10_7.txt'; 
% 
% writetable(T,filenamestore,'Delimiter','\t');
% 
% type processed_result_stereo_classical_2_7.txt;
% % type processed_result_stereo_classical_6_6.txt;
% % type processed_result_stereo_classical_8_6.txt;
% % type processed_result_stereo_classical_10_7.txt;
% 







