close all;
clear; 
clc;


%  filename1 = '3d_stereo_2_7.txt';
% filename1 = '3d_stereo_6_6.txt';
% filename1 = '3d_stereo_8_6.txt';
filename1 = '3d_stereo_10_7.txt';

%  filename2 = 'new_update_2_7.txt';
%  filename2 = 'new_update_6_6.txt';
%  filename2 = 'new_update_8_6.txt';
 filename2 = 'new_update_10_7.txt';

%  filename3 = 'classical_2_7.txt';
%  filename3 = 'classical_6_6.txt';
%  filename3 = 'classical_8_6.txt';
 filename3 = 'classical_10_7.txt';
 
f1 = fopen(filename1);
data = textscan(f1,'%s');
fclose(f1);
variable = str2double(data{1}(1:1:end));
three_d_stereo_resultsdata = zeros(8,4);

for i = 1:8
    
    three_d_stereo_resultsdata(i,1) = variable(4*i-3);
    three_d_stereo_resultsdata(i,2) = variable(4*i-2);
    three_d_stereo_resultsdata(i,3) = variable(4*i-1);
    three_d_stereo_resultsdata(i,4) = variable(4*i);
    
end


f2 = fopen(filename2);
data = textscan(f2,'%s');
fclose(f2);
variable = str2double(data{1}(1:1:end));
new_update_resultsdata = zeros(8,4);

for i = 1:8
    
    new_update_resultsdata(i,1) = variable(4*i-3);
    new_update_resultsdata(i,2) = variable(4*i-2);
    new_update_resultsdata(i,3) = variable(4*i-1);
    new_update_resultsdata(i,4) = variable(4*i);
    
end


f3 = fopen(filename3);
data = textscan(f3,'%s');
fclose(f3);
variable = str2double(data{1}(1:1:end));
classical_resultsdata = zeros(8,3);

for i = 1:8
    
    classical_resultsdata(i,1) = variable(3*i-2);
    classical_resultsdata(i,2) = variable(3*i-1);
    classical_resultsdata(i,3) = variable(3*i);
    
end


% Data Metrics for 3d_stereo and new_update resultsdata
% # radius 
% # no. of points 
% # maximum accuracy from 1- 50 iterations
% # iteration number at which maximum accuracy was achieved


% Data Metrics for classical resultsdata
% # no. of points 
% # maximum accuracy from 1- 50 iterations
% # iteration number at which maximum accuracy was achieved

%% it no gain vs num points 

figure 

semilogx(three_d_stereo_resultsdata(:,2),-three_d_stereo_resultsdata(:,4)+classical_resultsdata(:,3),'LineWidth',2.5)
hold on
semilogx(new_update_resultsdata(:,2),-new_update_resultsdata(:,4)+classical_resultsdata(:,3),'LineWidth',2.5)
hold on
grid on 
xlabel('Number of Points')
xlim([630 55000])
ylabel('Gain in Iteration Number for Maximum Accuracy')
ylim([-1.1 24.1])
% % xlim([2.5 5])
% 
legend('3D Stereographic','Quantum Analogue','Location','best')

% %% it no vs num points 
% 
% figure 
% 
% plot(log10(three_d_stereo_resultsdata(:,2)),three_d_stereo_resultsdata(:,4))
% hold on
% plot(log10(new_update_resultsdata(:,2)),new_update_resultsdata(:,4))
% hold on
% plot(log10(classical_resultsdata(:,1)),classical_resultsdata(:,3))
% hold on
% grid on 
% xlabel('log(Number of Points)')
% ylabel('Iteration Number for Maximum Accuracy')
% % ylim([0 7])
% % xlim([2.5 5])
% 
% legend('3D Stereographic','Quantum Analogue','Classical','Location','best')

% %%  it no vs num points 
% 
% figure 
% 
% 
% plot(log10(new_update_resultsdata(:,2)),new_update_resultsdata(:,4))
% hold on
% plot(log10(classical_resultsdata(:,1)),classical_resultsdata(:,3))
% hold on
% grid on 
% xlabel('log(Number of Points)')
% ylabel('Iteration Number for Maximum Accuracy')
% % ylim([0 7])
% % xlim([2.5 5])
% 
% legend('Quantum Analogue','Classical','Location','best')

%% max acc vs num pts

figure 

semilogx(three_d_stereo_resultsdata(:,2),three_d_stereo_resultsdata(:,3),'LineWidth',2.5)
hold on
semilogx(new_update_resultsdata(:,2),new_update_resultsdata(:,3),'LineWidth',2.5)
hold on
plot(classical_resultsdata(:,1),classical_resultsdata(:,2),'LineWidth',2.5)
hold on
grid on

xlabel('Number of Points')
xlim([630 55000])
ylabel('Maximum Accuracy(%)')
legend('3D Stereographic','Quantum Analogue','Classical','Location','best')
% ylim([82.2 87.9])
% 
% % %% max acc vs num pts
% % 
% % figure 
% % plot(log10(three_d_stereo_resultsdata(:,2)),three_d_stereo_resultsdata(:,3))
% % hold on
% % scatter(log10(three_d_stereo_resultsdata(:,2)),three_d_stereo_resultsdata(:,3), '^')
% % hold on
% % for k = 1:8
% %     
% %     hold on
% %     grid on
% %     toadd  = append("r=",string(three_d_stereo_resultsdata(k,1)),", it. no.=",string(three_d_stereo_resultsdata(k,4))); 
% % %     labels = append(labels,newline,toadd);
% %     labelpoints(log10(three_d_stereo_resultsdata(k,2)),three_d_stereo_resultsdata(k,3), toadd)
% % end
% % 
% % plot(log10(new_update_resultsdata(:,2)),new_update_resultsdata(:,3))
% % hold on
% % scatter(log10(new_update_resultsdata(:,2)),new_update_resultsdata(:,3),'^')
% % hold on
% % for k = 1:8
% %     
% %     hold on
% %     grid on
% %     toadd2  = append("r=",string(new_update_resultsdata(k,1)),", it. no.= ",string(new_update_resultsdata(k,4))); 
% % %     labels = append(labels,newline,toadd);
% %     labelpoints(log10(new_update_resultsdata(k,2)),new_update_resultsdata(k,3), toadd2)
% % end
% % 
% % plot(log10(classical_resultsdata(:,1)),classical_resultsdata(:,2))
% % hold on
% % scatter(log10(classical_resultsdata(:,1)),classical_resultsdata(:,2),'>')
% % hold on
% % for k = 1:8
% %     
% %     hold on
% %     grid on
% %     toadd5 = append("it. no.= ", string(classical_resultsdata(k,3))); 
% % %     labels = append(labels,newline,toadd);
% %     labelpoints(log10(classical_resultsdata(:,1)),classical_resultsdata(:,2), toadd5 )
% % end
% % hold on
% % grid on
% % 
% % xlabel('log(Number of Points)')
% % ylabel('Maximum Accuracy(%)')
% % legend('3D Stereographic','3D Stereographic','Quantum Analogue','Quantum Analogue','Classical','Classical','Location','best')
% 
%% max acc gain vs num pts
figure 
semilogx(three_d_stereo_resultsdata(:,2),three_d_stereo_resultsdata(:,3)-classical_resultsdata(:,2),'LineWidth',2.5)
hold on
semilogx(new_update_resultsdata(:,2),new_update_resultsdata(:,3)-classical_resultsdata(:,2),'LineWidth',2.5)
hold on
grid on

xlabel('Number of Points')
xlim([630 55000])
ylabel('Maximum Accuracy Gain(%)')
legend('3D Stereographic','Quantum Analogue','Location','best')
ylim([0 0.5])
% 
% %% max acc gain vs num pts
% figure 
% plot(log10(three_d_stereo_resultsdata(:,2)),three_d_stereo_resultsdata(:,3)-classical_resultsdata(:,2),'LineWidth',2.5)
% hold on
% scatter(log10(three_d_stereo_resultsdata(:,2)),three_d_stereo_resultsdata(:,3)-classical_resultsdata(:,2), '^')
% hold on
% for k = 1:8
%     
%     hold on
%     grid on
%     toadd3  = append("r=",string(three_d_stereo_resultsdata(k,1)),", it. no.=",string(three_d_stereo_resultsdata(k,4))); 
% %     labels = append(labels,newline,toadd);
%     labelpoints(log10(three_d_stereo_resultsdata(k,2)),three_d_stereo_resultsdata(k,3)-classical_resultsdata(k,2), toadd3)
% end
% 
% 
% hold on
% plot(log10(new_update_resultsdata(:,2)),new_update_resultsdata(:,3)-classical_resultsdata(:,2),'LineWidth',2.5)
% hold on
% scatter(log10(new_update_resultsdata(:,2)),new_update_resultsdata(:,3)-classical_resultsdata(:,2),'^')
% hold on
% for k = 1:8
%     
%     hold on
%     grid on
%     toadd4  = append("r=",string(new_update_resultsdata(k,1)),", it. no.=",string(new_update_resultsdata(k,4))); 
% %     labels = append(labels,newline,toadd);
%     labelpoints(log10(new_update_resultsdata(k,2)),new_update_resultsdata(k,3)-classical_resultsdata(k,2), toadd4)
% end
% 
% 
% hold on
% % plot(log10(classical_resultsdata(:,1)),classical_resultsdata(:,2))
% grid on
% 
% xlabel('log(Number of Points)')
% ylabel('Maximum Accuracy Gain(%)')
% legend('3D Stereographic','3D Stereographic','Quantum Analogue','Quantum Analogue','Location','best')
% ylim([0 0.5])
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
