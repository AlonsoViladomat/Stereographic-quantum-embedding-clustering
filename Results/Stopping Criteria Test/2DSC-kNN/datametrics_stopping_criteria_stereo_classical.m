close all;
clear; 
clc;


% filename = 'data_2_7.txt';
% filename = 'data_6_6.txt';
% filename = 'data_8_6.txt';
filename = 'data_10_7.txt';

f = fopen(filename);
data = textscan(f,'%s');
fclose(f);
variable = str2double(data{1}(1:1:end));
resultsdata = zeros(128,102);

for i = 1:128
    resultsdata(i,1) = variable(102*i-101);
    resultsdata(i,2) = variable(102*i-100);
    resultsdata(i,3:52) = variable(102*i-99:102*i-50);
    resultsdata(i,53:102) = variable(102*i-49:102*i);
    
end

% Data Metrics
% # radius 
% # no. of points 
% # maximum accuracy from 1- 50 iterations
% # iteration number at which maximum accuracy was achieved


datametric = zeros(128,4);

for j = 1:128
    

   accuracies = resultsdata(j,3:52);
    
   
   datametric(j,1) = resultsdata(j, 1);
   datametric(j,2) = resultsdata(j, 2);
   [datametric(j,3) , datametric(j,4)] = max(accuracies);
    
   
end

datametric_ahead = datametric; 

datametric(:,3) = round(datametric(:,3),3);
datametric(:,4) = round(datametric(:,4),3);

%% Surface plots 
iterations = 1:50;
labels = strings; 
C = ["#7E2F8E" ; "magenta" ; "cyan" ; "blue" ; "green" ; "yellow" ; "red"; "white" ];


% for iterator = 1:8 % for accuracy 
for iterator = 8  % for prob stopping
        
    toadd  = string(resultsdata(16*iterator-15,2)); 
    labels = append(labels,newline,toadd);

    radii = resultsdata(16*iterator-15 : 16*iterator , 1); 
%     surface = resultsdata(16*iterator-15 : 16*iterator,3:52) ; % for accuracy
    surface = resultsdata(16*iterator-15 : 16*iterator,53:102) ; % for probability of stopping
   
%     surf(iterations,radii,surface,'FaceAlpha',0.5);% useless??
    surf(iterations,radii,surface);
%     surf(iterations,radii,surface,'FaceColor',C(iterator)); % accuracy
    set(gca,'YScale','log')
 

    fontsize(gca,24,"pixels")
    
    hold on
end
% lgd = legend('640','1280','2560','3200','6400','12800','25600','51200')
% fontsize(lgd,16,'points')

xlabel('Iteration Number','FontSize',24)
% ylabel('Radius','FontSize',24)
zlabel('Probability of stopping','FontSize',22)
zlabel('Accuracy (%)','FontSize',24)
% zlim([20 90])
% zlim([85.4 87.2])
% ylim([1 100])
% xlim([0 50])
LX = get(gca,'XLim');
LY = get(gca,'YLim');
LZ = get(gca,'ZLim');
set(gca,'XTick',LX(1):10:LX(2))
% set(gca,'YTick',LY(1):0.5:LY(2))
set(gca,'ZTick',LZ(1):0.2:LZ(2))
% v = [143.275 6.84311131418182];
% view([81.3249999999999 18.6827277533017])
% view([ 68.9583333333334 16.2451776649746 ])
view([-37.5 30])


% %% Plotting  accuracy and probability of stopping for each radius and number of points
% 
% figure
% % labels = strings;
% for k = 1:1:128
%     
%     
%     plot(resultsdata(k,3:52))
%     hold on 
%     scatter(datametric(k,4), datametric(k,3), 'O')
%     hold on
%     grid on
%     toadd  = append(string(datametric(k,1))," ",string(datametric(k,2))); 
% %     labels = append(labels,newline,toadd);
%     labelpoints(datametric(k,4), datametric(k,3), toadd)
%     
% 
% end
% %legend(legends)
% xlabel('Iteration Number')
% ylabel('Accuracy')
% 
% 
% figure
% 
% labels = strings;
% for l = 1:128
%     
%     
%     plot(resultsdata(l,53:102))
%     hold on
% %     toadd  = append(string(datametric(l,1))," ",string(datametric(l,2))); 
% %     labels = append(labels,newline,toadd);
% 
%     hold on
%     grid on
% 
% end
% 
% xlabel('Iteration Number')
% ylabel('Probabilty of stopping')
% legend(labels,'Location','best','NumColumns',16)

% %% storing complete processed result 
%  filenamestore = 'complete_processed_result_stopping_criteria_new_update_short_2_7.txt'; 
% % filenamestore = 'complete_processed_result_stopping_criteria_new_update_short_6_6.txt'; 
% % filenamestore = 'complete_processed_result_stopping_criteria_new_update_short_8_6.txt'; 
% % filenamestore = 'complete_processed_result_stopping_criteria_new_update_short_10_7.txt'; 
% 
% writematrix(datametric,filenamestore,'Delimiter','\t');  
%  type complete_processed_result_stopping_criteria_new_update_short_2_7.txt;
% % type complete_processed_result_stopping_criteria_new_update_short_6_6.txt;
% % type complete_processed_result_stopping_criteria_new_update_short_8_6.txt;
% % type complete_processed_result_stopping_criteria_new_update_short_10_7.txt;
% 
%  xlfilenamestore = 'complete_processed_result_stopping_criteria_new_update_short_2_7.xlsx'; 
% % xlfilenamestore = 'complete_processed_result_stopping_criteria_new_update_short_6_6.xlsx'; 
% % xlfilenamestore = 'complete_processed_result_stopping_criteria_new_update_short_8_6.xlsx'; 
% % xlfilenamestore = 'complete_processed_result_stopping_criteria_new_update_short_10_7.xlsx'; 
% 
% xlswrite(xlfilenamestore,datametric);
% 
% 
% %% Further processing of results
% 
% datametric_acc = datametric_ahead;
% 
% for k = 1:8
%     
%    index3 = 16*(k-1) + 1;
%    index4 = 16*k;
%    A = datametric_acc(index3:index4, :);
%    A = sortrows(A,3,'descend');
%    datametric_acc(index3:index4, :) = A;
%    
% end
% 
% 
% datametric_acc(:,3) = round(datametric_acc(:,3),3);
% datametric_acc(:,4) = round(datametric_acc(:,4),3);
% 
% 
% winner_acc = zeros(8,4);
% 
% for m = 1:8
%     index2 = 16*(m-1)+1;
%     winner_acc(m,:) = datametric_acc(index2,:);
% end
% 
% %% storing processed result sorted by accuracy
%  filenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_2_7.txt'; 
% % filenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_6_6.txt'; 
% % filenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_8_6.txt'; 
% % filenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_10_7.txt'; 
% % 
% writematrix(winner_acc,filenamestore,'Delimiter','\t');  
%  type accuracy_processed_result_stopping_criteria_new_update_short_2_7.txt;
% % type accuracy_processed_result_stopping_criteria_new_update_short_6_6.txt;
% % type accuracy_processed_result_stopping_criteria_new_update_short_8_6.txt;
% % type accuracy_processed_result_stopping_criteria_new_update_short_10_7.txt;
% 
%  xlfilenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_2_7.xlsx'; 
% % xlfilenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_6_6.xlsx'; 
% % xlfilenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_8_6.xlsx'; 
% % xlfilenamestore = 'accuracy_processed_result_stopping_criteria_new_update_short_10_7.xlsx'; 
% 
% xlswrite(xlfilenamestore,winner_acc);
% 
% 
