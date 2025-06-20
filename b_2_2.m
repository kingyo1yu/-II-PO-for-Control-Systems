%% 加热炉闭环控制系统（Cohen-Coon、PSO与GA对比）
clear all; close all; clc;

% ========== 1. 系统参数 ==========
K = 9.8519;           % 被控对象增益
T = 2853.2284;        % 时间常数 (s)
tau = 184.4717;       % 纯滞后时间 (s)
setpoint = 35;        % 设定温度 (°C)
room_temp = 16.8;     % 室温 (°C)
sim_time = 8000;      % 仿真时间 (s)
dt = 1;               % 仿真步长 (s)
t = 0:dt:sim_time;    % 时间向量

% ========== 2. PID 控制逻辑（通用函数） ==========
function [y, u, error] = pid_control(t, Kp, Ki, Kd, setpoint, room_temp, K_sys, T_sys, tau_sys, dt)
    n = length(t);
    y = zeros(n, 1);   % 温度输出
    u = zeros(n, 1);   % 控制输入（电压）
    error = zeros(n, 1); % 误差
    integral = 0;      % 积分项
    prev_error = 0;    % 上一时刻误差
    
    y(1) = room_temp;  % 初始温度为室温
    delay_steps = round(tau_sys / dt);  % 时滞步数
    u_queue = zeros(1, delay_steps);  % 控制输入队列
    
    for i = 2:n
        % 计算误差
        error(i) = setpoint - y(i-1);
        
        % 积分项（带抗饱和）
        integral = integral + error(i)*dt;
        integral = max(min(integral, 10/Ki), -10/Ki);  % 积分限幅
        
        % 微分项
        if i > 2
            derivative = (error(i) - prev_error)/dt;
        else
            derivative = 0;
        end
        prev_error = error(i);
        
        % PID 输出
        u(i) = Kp*error(i) + Ki*integral + Kd*derivative;
        u(i) = max(min(u(i), 10), 0);  % 电压限幅 [0, 10]
        
        % 纯滞后补偿
        u_queue = [u(i), u_queue(1:end-1)];
        u_delayed = u_queue(end);
        
        % 系统动态响应
        y(i) = y(i-1) + ((-y(i-1) + room_temp + K_sys*u_delayed)/T_sys)*dt;
    end
end % 结束pid_control函数

% ========== 3. Cohen-Coon方法计算PID参数 ==========
function [Kp, Ki, Kd] = cohen_coon_pid(K, T, tau)
    % Cohen-Coon公式：适用于带纯滞后的一阶系统
    theta = tau / T;  % 时滞比
    
    % 比例系数
    Kp = (T / (K * tau)) * (0.9 + 0.2 * theta) / (1 + 3.3 * theta);
    
    % 积分时间
    Ti = tau * (3.3 + 3.6 * theta) / (1 + 0.7 * theta);
    
    % 微分时间
    Td = tau * (0.5 + 0.3 * theta) / (1 + 0.6 * theta);
    
    % 转换为Ki、Kd
    Ki = Kp / Ti;
    Kd = Kp * Td;
end % 结束cohen_coon_pid函数

% ========== 4. 遗传算法优化PID参数 ==========
function [best_params, best_fitness] = ga_pid_optimization(t, setpoint, room_temp, K, T, tau, dt)
    % GA参数
    pop_size = 80;          % 种群大小
    num_generations = 150;  % 迭代次数
    num_genes = 3;          % 参数维度 (Kp, Ki, Kd)
    crossover_rate = 0.8;   % 交叉率
    mutation_rate = 0.1;    % 变异率
    
    % 参数范围
    param_ranges = [
        1.0, 30;          % Kp范围
        1e-6, 5e-4;       % Ki范围
        50, 500           % Kd范围
    ];
    
    % 初始化种群
    population = zeros(pop_size, num_genes);
    for i = 1:pop_size
        for j = 1:num_genes
            population(i, j) = param_ranges(j, 1) + rand() * (param_ranges(j, 2) - param_ranges(j, 1));
        end
    end
    
    % 记录每代最优解
    best_fitness_history = zeros(num_generations, 1);
    best_params_history = zeros(num_generations, num_genes);
    
    % 主循环
    for gen = 1:num_generations
        % 评估适应度
        fitness = zeros(pop_size, 1);
        for i = 1:pop_size
            fitness(i) = evaluate_fitness(population(i, :), t, setpoint, room_temp, K, T, tau, dt);
        end
        
        % 记录当前代的最优解
        [max_fitness, max_idx] = max(fitness);
        best_fitness_history(gen) = max_fitness;
        best_params_history(gen, :) = population(max_idx, :);
        
        % 选择操作 - 轮盘赌选择
        new_population = zeros(pop_size, num_genes);
        total_fitness = sum(fitness);
        selection_probs = fitness / total_fitness;
        
        for i = 1:pop_size
            % 轮盘赌选择父代
            r = rand();
            cumulative_prob = 0;
            for j = 1:pop_size
                cumulative_prob = cumulative_prob + selection_probs(j);
                if cumulative_prob >= r
                    new_population(i, :) = population(j, :);
                    break;
                end
            end
        end
        
        % 交叉操作
        for i = 1:2:pop_size-1
            if rand() < crossover_rate
                % 算术交叉
                alpha = rand();
                temp1 = alpha * new_population(i, :) + (1-alpha) * new_population(i+1, :);
                temp2 = (1-alpha) * new_population(i, :) + alpha * new_population(i+1, :);
                
                % 确保参数在范围内
                for j = 1:num_genes
                    temp1(j) = max(min(temp1(j), param_ranges(j, 2)), param_ranges(j, 1));
                    temp2(j) = max(min(temp2(j), param_ranges(j, 2)), param_ranges(j, 1));
                end
                
                new_population(i, :) = temp1;
                new_population(i+1, :) = temp2;
            end
        end
        
        % 变异操作
        for i = 1:pop_size
            for j = 1:num_genes
                if rand() < mutation_rate
                    % 高斯变异
                    sigma = 0.1 * (param_ranges(j, 2) - param_ranges(j, 1));
                    new_population(i, j) = new_population(i, j) + sigma * randn();
                    
                    % 确保参数在范围内
                    new_population(i, j) = max(min(new_population(i, j), param_ranges(j, 2)), param_ranges(j, 1));
                end
            end
        end
        
        % 精英保留
        [~, worst_idx] = min(fitness);
        new_population(worst_idx, :) = best_params_history(gen, :);
        
        % 更新种群
        population = new_population;
        
        % 输出迭代信息
        if mod(gen, 10) == 0
            fprintf('GA迭代 %d: 最优适应度 = %.4f, 参数 = [%.4f, %.6f, %.4f]\n', ...
                gen, max_fitness, best_params_history(gen, 1), best_params_history(gen, 2), best_params_history(gen, 3));
        end
    end
    
    % 找到全局最优解
    [best_fitness, best_idx] = max(best_fitness_history);
    best_params = best_params_history(best_idx, :);
    
    % 绘制适应度进化曲线
    figure;
    plot(1:num_generations, best_fitness_history, 'LineWidth', 2);
    xlabel('迭代次数');
    ylabel('适应度值');
    title('GA优化过程');
    grid on;
end % 结束ga_pid_optimization函数

% ========== 5. 评估适应度函数 ==========
function fitness = evaluate_fitness(params, t, setpoint, room_temp, K, T, tau, dt)
    Kp = params(1);
    Ki = params(2);
    Kd = params(3);
    
    % 运行PID控制仿真
    [y, ~, error] = pid_control(t, Kp, Ki, Kd, setpoint, room_temp, K, T, tau, dt);
    
    % 计算IAE
    iae = trapz(t, abs(error));
    
    % 计算调节时间
    tolerance = 0.02*setpoint;
    settling_time = calculate_settling_time(error, t, tolerance);
    
    % 计算超调量
    overshoot = max(y) - setpoint;
    overshoot = max(overshoot, 0);
    
    % 计算稳态误差
    steady_idx = round(0.9*length(t)):length(t);
    steady_error = mean(abs(error(steady_idx)));
    
    % 综合性能指标（加权组合，根据需求调整权重）
    weight_iae = 1.0;
    weight_settling = 0.1;
    weight_overshoot = 2.0;
    weight_steady = 2.0;
    
    performance = weight_iae * iae + weight_settling * settling_time + weight_overshoot * overshoot + weight_steady * steady_error;
    
    % 将目标函数转换为适应度（越小越好，所以取倒数）
    fitness = 1 / (1 + performance);
end % 结束evaluate_fitness函数

% ========== 6. 计算调节时间 ==========
function settling_time = calculate_settling_time(error, t, tolerance)
    % 寻找误差首次低于容差的点
    first_crossing = find(abs(error) < tolerance, 1, 'first');
    
    if ~isempty(first_crossing)
        % 检查从该点开始到结束，误差是否一直保持在容差范围内
        remaining_errors = abs(error(first_crossing:end));
        if all(remaining_errors < tolerance)
            settling_time = t(first_crossing);
        else
            % 寻找最后一个稳定区域
            last_stable_start = 0;
            current_length = 0;
            max_length = 0;
            
            for i = 1:length(error)
                if abs(error(i)) < tolerance
                    current_length = current_length + 1;
                    if current_length > max_length
                        max_length = current_length;
                        last_stable_start = i - max_length + 1;
                    end
                else
                    current_length = 0;
                end
            end
            
            if last_stable_start > 0
                settling_time = t(last_stable_start);
            else
                settling_time = max(t);
            end
        end
    else
        settling_time = max(t);
    end
end % 结束calculate_settling_time函数

% ========== 7. 性能指标计算 ==========
function [rmse, mae, steady_error, overshoot, settling_time] = calc_performance(y, error, setpoint, sim_time, t, dt)
    % 均方根误差
    rmse = sqrt(mean(error.^2));
    
    % 平均绝对误差
    mae = mean(abs(error));
    
    % 稳态误差（最后10%时间）
    steady_idx = round(0.9*length(t)):length(t);
    steady_error = mean(error(steady_idx));
    
    % 超调量
    overshoot = max(y) - setpoint;
    overshoot = max(overshoot, 0);
    
    % 调节时间
    tolerance = 0.02*setpoint;
    settling_time = calculate_settling_time(error, t, tolerance);
end % 结束calc_performance函数

% ========== 8. 主程序：Cohen-Coon、PSO与GA对比 ==========
% 1. Cohen-Coon方法计算PID参数
[Kp_cc, Ki_cc, Kd_cc] = cohen_coon_pid(K, T, tau);
fprintf('Cohen-Coon整定的PID参数:\n');
fprintf('Kp = %.4f, Ki = %.6f, Kd = %.4f\n', Kp_cc, Ki_cc, Kd_cc);

% 2. 运行Cohen-Coon参数仿真
[y_cc, u_cc, error_cc] = pid_control(t, Kp_cc, Ki_cc, Kd_cc, setpoint, room_temp, K, T, tau, dt);
[rmse_cc, mae_cc, steady_error_cc, overshoot_cc, settling_time_cc] = calc_performance(y_cc, error_cc, setpoint, sim_time, t, dt);
fprintf('Cohen-Coon控制效果:\n');
fprintf('RMSE = %.4f, MAE = %.4f, 稳态误差 = %.4f, 超调量 = %.4f, 调节时间 = %.2f秒\n', ...
    rmse_cc, mae_cc, steady_error_cc, overshoot_cc, settling_time_cc);

% 3. GA优化PID参数
fprintf('\n开始GA优化...\n');
[best_params_ga, best_fitness_ga] = ga_pid_optimization(t, setpoint, room_temp, K, T, tau, dt);
Kp_ga = best_params_ga(1);
Ki_ga = best_params_ga(2);
Kd_ga = best_params_ga(3);
fprintf('GA优化的PID参数:\n');
fprintf('Kp = %.4f, Ki = %.6f, Kd = %.4f, 适应度 = %.4f\n', Kp_ga, Ki_ga, Kd_ga, best_fitness_ga);

% 4. 运行GA参数仿真
[y_ga, u_ga, error_ga] = pid_control(t, Kp_ga, Ki_ga, Kd_ga, setpoint, room_temp, K, T, tau, dt);
[rmse_ga, mae_ga, steady_error_ga, overshoot_ga, settling_time_ga] = calc_performance(y_ga, error_ga, setpoint, sim_time, t, dt);
fprintf('GA控制效果:\n');
fprintf('RMSE = %.4f, MAE = %.4f, 稳态误差 = %.4f, 超调量 = %.4f, 调节时间 = %.2f秒\n', ...
    rmse_ga, mae_ga, steady_error_ga, overshoot_ga, settling_time_ga);

% 5. 可视化对比
figure('Position', [100, 100, 1200, 800]);

% 温度响应对比
subplot(2,2,1);
plot(t, y_cc, 'b-', t, y_ga, 'g-', t, setpoint*ones(size(t)), 'k--', 'LineWidth', 2);
xlabel('时间 (秒)'); ylabel('温度 (°C)');
title('温度响应对比');
legend('Cohen-Coon', 'GA', '设定温度');
grid on;

% 误差对比
subplot(2,2,2);
plot(t, error_cc, 'b-', t, error_ga, 'g-', 'LineWidth', 2);
xlabel('时间 (秒)'); ylabel('误差 (°C)');
title('误差曲线对比');
grid on;

% 控制量对比
subplot(2,2,3);
plot(t, u_cc, 'b-', t, u_ga, 'g-', 'LineWidth', 2);
xlabel('时间 (秒)'); ylabel('电压 (V)');
title('控制输入对比');
grid on;

% 性能指标表格
subplot(2,2,4);
cla;
text(0.1, 0.9, '性能指标对比', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.7, ['RMSE (°C):', newline, ...
    'Cohen-Coon: ', num2str(rmse_cc, '%.4f'), newline, ...
    'GA: ', num2str(rmse_ga, '%.4f')], 'FontSize', 12);
text(0.5, 0.7, ['调节时间 (秒):', newline, ...
    'Cohen-Coon: ', num2str(settling_time_cc, '%.2f'), newline, ...
    'GA: ', num2str(settling_time_ga, '%.2f')], 'FontSize', 12);
text(0.1, 0.3, ['超调量 (°C):', newline, ...
    'Cohen-Coon: ', num2str(overshoot_cc, '%.4f'), newline, ...
    'GA: ', num2str(overshoot_ga, '%.4f')], 'FontSize', 12);
text(0.5, 0.3, ['稳态误差 (°C):', newline, ...
    'Cohen-Coon: ', num2str(steady_error_cc, '%.4f'), newline, ...
    'GA: ', num2str(steady_error_ga, '%.4f')], 'FontSize', 12);
axis([0 1 0 1]);
set(gca, 'Visible', 'off');