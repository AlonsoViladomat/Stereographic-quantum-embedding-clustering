close all;
clear; 
clc;

 filename = 'data_2_7.txt';
% filename = 'data_6_6.txt';
% filename = 'data_8_6.txt';
% filename = 'data_10_7.txt';
f = fopen(filename);
data = textscan(f,'%s');
fclose(f);
variable = str2double(data{1}(1:1:end));
resultsdata = zeros(800,8);

for i = 1:800
    resultsdata(i,1) = variable(8*i-7);
    resultsdata(i,2) = variable(8*i-6);
    resultsdata(i,3) = variable(8*i-5);
    resultsdata(i,4) = variable(8*i-4);
    resultsdata(i,5) = variable(8*i-3);
    resultsdata(i,6) = variable(8*i-2);
    resultsdata(i,7) = variable(8*i-1);
    resultsdata(i,8) = variable(8*i);
end

% Data Metrics
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

datametric = zeros(8,14);

for j = 1:8
    
   index1 = 100*(j-1) + 1;
   index2 = 100*j;
   
   diff_acc = resultsdata(index1:index2, 2);
   training_acc = resultsdata(index1:index2, 3);
   testing_acc = resultsdata(index1:index2, 4);
   training_num_iter = resultsdata(index1:index2, 5);
   training_exec_time = resultsdata(index1:index2, 7);
   testing_num_iter = resultsdata(index1:index2, 6);
   testing_exec_time = resultsdata(index1:index2, 8);
   
   datametric(j,1) = resultsdata(index1, 1);
   datametric(j,2) = mean(diff_acc);
   datametric(j,3) = mean(training_acc);
   datametric(j,4) = std(training_acc);
   datametric(j,5) = mean(testing_acc);
   datametric(j,6) = std(testing_acc);
   datametric(j,7) = mean(training_num_iter);
   datametric(j,8) = std(training_num_iter);
   datametric(j,9) = mean(testing_num_iter);
   datametric(j,10) = std(testing_num_iter);
   datametric(j,11) = mean(training_exec_time);
   datametric(j,12) = std(training_exec_time);
   datametric(j,13) = mean(testing_exec_time);
   datametric(j,14) = std(testing_exec_time);
   
end

datametric(:,2) = round(datametric(:,2),3);
datametric(:,3) = round(datametric(:,3),3);
datametric(:,4) = round(datametric(:,4),3);
datametric(:,5) = round(datametric(:,5),3);
datametric(:,6) = round(datametric(:,6),3);
datametric(:,7) = round(datametric(:,7),3);
datametric(:,8) = round(datametric(:,8),3);
datametric(:,9) = round(datametric(:,9),3);
datametric(:,10) = round(datametric(:,10),3);
datametric(:,11) = round(1000*datametric(:,11),3); %storing time in milliseconds
datametric(:,12) = round(1000*datametric(:,12),3); %storing time in milliseconds
datametric(:,13) = round(1000*datametric(:,13),3); %storing time in milliseconds
datametric(:,14) = round(1000*datametric(:,14),3); %storing time in milliseconds
%% processed result

 xlfilenamestore = 'processed_result_overfitting_test_classical_2_7.xlsx';
% xlfilenamestore = 'processed_result_overfitting_test_classical_6_6.xlsx';
% xlfilenamestore = 'processed_result_overfitting_test_classical_8_6.xlsx';
% xlfilenamestore = 'processed_result_overfitting_test_classical_10_7.xlsx';

xlswrite(xlfilenamestore,datametric);


 filenamestore = 'processed_result_overfitting_test_classical_2_7.txt'; 
% filenamestore = 'processed_result_overfitting_test_classical_6_6.txt'; 
% filenamestore = 'processed_result_overfitting_test_classical_8_6.txt'; 
% filenamestore = 'processed_result_overfitting_test_classical_10_7.txt'; 

 writematrix(datametric,filenamestore,'Delimiter','\t')  
  type processed_result_overfitting_test_classical_2_7.txt;
% type processed_result_overfitting_test_classical_6_6.txt;
% type processed_result_overfitting_test_classical_8_6.txt;
% type processed_result_overfitting_test_classical_10_7.txt;









