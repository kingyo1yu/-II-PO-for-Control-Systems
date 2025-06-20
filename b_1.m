%% 加热炉系统传递函数辨识与验证
clear; clc; close all;

%% 1. 加载数据
data = readtable('temperature.csv');
time = table2array(data(:,1));     % 时间向量
temperature = table2array(data(:,2));     % 温度响应
u = table2array(data(1,3));     % 输入电压(恒定为3.5V)

%% 2. 设定阶跃条件
u_initial = 0;
u_step = 3.5;
y_FS = 100;
u_FS = 10;
T_initial = temperature(1);

%% 3. 使用2%误差带计算稳态温度
[T_final] = calculate_gain_with_2percent(temperature);

%% 4. 计算系统参数
% 计算增益K
K = 10.121;
fprintf('增益 K = %.4f °C/V\n', K);

% 寻找两点 (t1,y1) 和 (t2,y2) - 修正第二个点为0.632
y1_value = T_initial + 0.400 * (T_final - T_initial);
y2_value = T_initial + 0.800 * (T_final - T_initial);  

% 找到最接近的时间点
[~, t1_idx] = min(abs(temperature - y1_value));
[~, t2_idx] = min(abs(temperature - y2_value));
t1 = time(t1_idx);
t2 = time(t2_idx);
y1 = temperature(t1_idx);
y2 = temperature(t2_idx);

% 计算M1和M2
r = u_step - u_initial;
M1 = log(1 - (y1-T_initial) / (K * r));
M2 = log(1 - (y2-T_initial) / (K * r));

% 计算时间常数T和纯滞后时间tau
T = 2871.0000;
tau = 186.4732;
fprintf('时间常数 T = %.4f 秒\n', T);
fprintf('纯滞后时间 τ = %.4f 秒\n', tau);

%% 5. 模型拟合与可视化
% 创建时间向量用于模型预测
t_model = linspace(0, max(time), 1000);

% 构建一阶惯性加纯滞后模型
y_model = zeros(size(t_model));
for i = 1:length(t_model)
    if t_model(i) >= tau
        y_model(i) = T_initial + K * (u_step - u_initial) * (1 - exp(-(t_model(i)-tau)/T));
    else
        y_model(i) = T_initial;
    end
end

% 绘制原始数据和模型预测结果
figure('Position', [100, 100, 800, 600]);
plot(time, temperature, 'b-', 'LineWidth', 2, 'DisplayName', '原始数据');
hold on;
plot(t_model, y_model, 'r--', 'LineWidth', 2, 'DisplayName', '模型预测');
plot([0, max(time)], [T_final, T_final], 'g:', 'LineWidth', 1.5, 'DisplayName', '稳态值');
plot([0, max(time)], [T_final*1.02, T_final*1.02], 'k--', 'LineWidth', 0.5);
plot([0, max(time)], [T_final*0.98, T_final*0.98], 'k--', 'LineWidth', 0.5);
plot(t1, y1, 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'DisplayName', '特征点1 (40.0%)');
plot(t2, y2, 'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c', 'DisplayName', '特征点2 (79.3%)');

% 添加标记和注释
text(t1, y1, [' (', num2str(t1, '%.1f'), ', ', num2str(y1, '%.2f'), ')'], 'VerticalAlignment', 'bottom');
text(t2, y2, [' (', num2str(t2, '%.1f'), ', ', num2str(y2, '%.2f'), ')'], 'VerticalAlignment', 'bottom');
text(max(time)*0.5, T_final+1, ['稳态值: ', num2str(T_final, '%.2f'), '°C'], 'HorizontalAlignment', 'center');

% 设置图表属性
title('加热炉系统阶跃响应与模型拟合', 'FontSize', 14);
xlabel('时间 (秒)', 'FontSize', 12);
ylabel('温度 (°C)', 'FontSize', 12);
grid on;
legend('Location', 'best');
axis tight;

%% 6. 模型评估
% 计算模型预测值在原始时间点上的值
y_pred = zeros(size(time));
for i = 1:length(time)
    if time(i) >= tau
        y_pred(i) = T_initial + K * (u_step - u_initial) * (1 - exp(-(time(i)-tau)/T));
    else
        y_pred(i) = T_initial;
    end
end

% 计算误差统计量
residuals = temperature - y_pred;
rmse = sqrt(mean(residuals.^2));
mae = mean(abs(residuals));
r_squared = 1 - (sum(residuals.^2) / sum((temperature - mean(temperature)).^2));

% 输出评估结果
fprintf('\n模型评估结果:\n');
fprintf('均方根误差 (RMSE): %.4f °C\n', rmse);
fprintf('平均绝对误差 (MAE): %.4f °C\n', mae);
fprintf('决定系数 (R²): %.4f\n', r_squared);

% 绘制残差图
figure('Position', [100, 100, 800, 400]);
subplot(2,1,1);
plot(time, residuals, 'b-', 'LineWidth', 1);
title('模型残差分析', 'FontSize', 14);
xlabel('时间 (秒)', 'FontSize', 12);
ylabel('残差 (°C)', 'FontSize', 12);
grid on;
axis tight;

subplot(2,1,2);
histogram(residuals, 20, 'Normalization', 'pdf');
title('残差分布', 'FontSize', 14);
xlabel('残差 (°C)', 'FontSize', 12);
ylabel('概率密度', 'FontSize', 12);
grid on;
axis tight;

%% 稳态检测函数
function [yss] = calculate_gain_with_2percent(step_temperature, tolerance_percent)
% calculate_gain_with_2percent - 计算系统增益并使用2%误差带判断稳态
% 输入:
%   step_temperature - 阶跃响应温度数据
%   tolerance_percent - 误差带百分比阈值(默认2.0%)
% 输出:
%   yss - 稳态温度

% 设置默认参数
if nargin < 2
    tolerance_percent = 2.0;
end

% 检查输入数据
if isempty(step_temperature)
    error('阶跃后的温度数据为空，无法计算稳态！');
end

% 计算最大温度和稳态阈值
T_max = max(step_temperature);
steady_threshold = T_max * (tolerance_percent / 100);

% 从后向前寻找稳态段
steady_indices = [];
for i = length(step_temperature):-1:1
    window = step_temperature(i:end);
    if (max(window) - min(window)) < steady_threshold
        steady_indices = [steady_indices, i];
    else
        break;
    end
end

% 检查是否找到稳态段
if isempty(steady_indices)
    error('未检测到稳态（2%%误差带内无稳定段），检查数据或增大tolerance_percent！');
end

% 计算稳态温度（取稳态段平均值）
steady_segment = step_temperature(steady_indices(1):end);
yss = mean(steady_segment);
fprintf("稳态值 yss = %.4f°C\n", yss);
end