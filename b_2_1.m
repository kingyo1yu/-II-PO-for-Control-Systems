%% 加热炉闭环控制系统（Cohen-Coon与PSO对比）
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

% ========== 4. 粒子群优化 (PSO) 算法 ==========
function [gbest, gbest_obj] = pso_pid_optimization(t, setpoint, room_temp, K, T, tau, dt)
    % PSO参数
    N = 80;           % 粒子数
    D = 3;            % 参数维度 (Kp, Ki, Kd)
    max_iter = 150;   % 迭代次数
    c1 = 2; c2 = 2;   % 学习因子
    w_max = 0.9; w_min = 0.4; % 惯性权重
    
    % 参数范围（扩大Kp搜索范围）
    lb = [1.0, 5e-6, 2];   % Kp, Ki, Kd 下界
    ub = [30, 5e-4, 300];  % Kp, Ki, Kd 上界
    
    % 初始化
    particles = rand(N, D) .* (ub - lb) + lb;
    velocities = rand(N, D) .* (ub - lb) + lb;
    pbest = particles;
    pbest_obj = ones(N, 1) * inf;
    gbest = particles(1, :);
    gbest_obj = inf;
    
    for iter = 1:max_iter
        % 非线性惯性权重
        w = w_max - (w_max - w_min) * (iter / max_iter).^2;
        
        for i = 1:N
            % 计算目标函数（IAE）
            obj_val = calculate_iae(particles(i, :), t, setpoint, room_temp, K, T, tau, dt);
            
            % 更新最优解
            if obj_val < pbest_obj(i)
                pbest_obj(i) = obj_val;
                pbest(i, :) = particles(i, :);
            end
            if obj_val < gbest_obj
                gbest_obj = obj_val;
                gbest = particles(i, :);
            end
            
            % 更新速度和位置
            r1 = rand(1, D); r2 = rand(1, D);
            velocities(i, :) = w*velocities(i, :) + c1*r1.*(pbest(i, :) - particles(i, :)) + c2*r2.*(gbest - particles(i, :));
            particles(i, :) = particles(i, :) + velocities(i, :);
            particles(i, :) = max(min(particles(i, :), ub), lb);
        end
        
        % 输出迭代信息（每10次输出一次）
        if mod(iter, 10) == 0
            fprintf('PSO迭代 %d: 最优IAE = %.4f, 参数 = [%.4f, %.6f, %.4f]\n', ...
                iter, gbest_obj, gbest(1), gbest(2), gbest(3));
        end
    end
end % 结束pso_pid_optimization函数

% ========== 5. 计算IAE（积分绝对误差） ==========
function iae = calculate_iae(params, t, setpoint, room_temp, K, T, tau, dt)
    Kp = params(1);
    Ki = params(2);
    Kd = params(3);
    [~, ~, error] = pid_control(t, Kp, Ki, Kd, setpoint, room_temp, K, T, tau, dt);
    iae = trapz(t, abs(error));  % 积分绝对误差
end % 结束calculate_iae函数

% ========== 6. 性能指标计算 ==========
function [rmse, mae, steady_error, overshoot, settling_time] = calc_performance(y, error, setpoint, sim_time, t, dt)
    % 检查输入参数
    if nargin < 6
        error('calc_performance 函数需要至少6个输入参数');
    end
    
    % 均方根误差
    if ~isempty(error)
        rmse = sqrt(mean(error.^2));
    else
        rmse = 0;
    end
    
    % 平均绝对误差
    if ~isempty(error)
        mae = mean(abs(error));
    else
        mae = 0;
    end
    
    % 稳态误差（最后10%时间）
    if ~isempty(error) && length(error) > 10
        steady_idx = round(0.9*length(t)):length(t);
        steady_error = mean(error(steady_idx));
    else
        steady_error = 0;
    end
    
    % 超调量
    if ~isempty(y)
        overshoot = max(y) - setpoint;
        overshoot = max(overshoot, 0);
    else
        overshoot = 0;
    end
    
    % 调节时间（误差<2%设定值，并保持在该范围内）
    tolerance = 0.02*setpoint;
    
    if ~isempty(error)
        % 寻找误差首次低于容差的点
        first_crossing = find(abs(error) < tolerance, 1, 'first');
        
        if ~isempty(first_crossing)
            % 检查从该点开始到结束，误差是否一直保持在容差范围内
            remaining_errors = abs(error(first_crossing:end));
            if all(remaining_errors < tolerance)
                % 如果后续所有误差都满足条件，这就是调节时间
                settling_time = t(first_crossing);
            else
                % 否则，寻找最后一个持续满足条件的区域
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
                    settling_time = sim_time;
                end
            end
        else
            % 如果整个仿真过程中误差都未低于容差，使用仿真时间
            settling_time = sim_time;
        end
    else
        settling_time = sim_time;
    end
end

% ========== 7. 主程序：Cohen-Coon与PSO对比 ==========
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

% 3. PSO优化PID参数
fprintf('\n开始PSO优化...\n');
[gbest_pso, gbest_obj_pso] = pso_pid_optimization(t, setpoint, room_temp, K, T, tau, dt);
Kp_pso = gbest_pso(1);
Ki_pso = gbest_pso(2);
Kd_pso = gbest_pso(3);
fprintf('PSO优化的PID参数:\n');
fprintf('Kp = %.4f, Ki = %.6f, Kd = %.4f, IAE = %.4f\n', Kp_pso, Ki_pso, Kd_pso, gbest_obj_pso);

% 4. 运行PSO参数仿真
[y_pso, u_pso, error_pso] = pid_control(t, Kp_pso, Ki_pso, Kd_pso, setpoint, room_temp, K, T, tau, dt);
[rmse_pso, mae_pso, steady_error_pso, overshoot_pso, settling_time_pso] = calc_performance(y_pso, error_pso, setpoint, sim_time, t, dt);
fprintf('PSO控制效果:\n');
fprintf('RMSE = %.4f, MAE = %.4f, 稳态误差 = %.4f, 超调量 = %.4f, 调节时间 = %.2f秒\n', ...
    rmse_pso, mae_pso, steady_error_pso, overshoot_pso, settling_time_pso);

% 5. 可视化对比
figure('Position', [100, 100, 1200, 800]);

% 温度响应对比
subplot(2,2,1);
plot(t, y_cc, 'b-', t, y_pso, 'r-', 'LineWidth', 2);
hold on;
plot(t, setpoint*ones(size(t)), 'k--', 'LineWidth', 1.5);
xlabel('时间 (秒)'); ylabel('温度 (°C)');
title('温度响应对比');
legend('Cohen-Coon', 'PSO', '设定温度');
grid on;

% 误差对比
subplot(2,2,2);
plot(t, error_cc, 'b-', t, error_pso, 'r-', 'LineWidth', 2);
xlabel('时间 (秒)'); ylabel('误差 (°C)');
title('误差曲线对比');
grid on;

% 控制量对比
subplot(2,2,3);
plot(t, u_cc, 'b-', t, u_pso, 'r-', 'LineWidth', 2);
xlabel('时间 (秒)'); ylabel('电压 (V)');
title('控制输入对比');
grid on;

% 性能指标表格
subplot(2,2,4);
cla;
text(0.1, 0.9, '性能指标对比', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.7, ['RMSE (°C):', newline, ...
    'Cohen-Coon: ', num2str(rmse_cc, '%.4f'), newline, ...
    'PSO: ', num2str(rmse_pso, '%.4f')], 'FontSize', 12);
text(0.5, 0.7, ['调节时间 (秒):', newline, ...
    'Cohen-Coon: ', num2str(settling_time_cc, '%.2f'), newline, ...
    'PSO: ', num2str(settling_time_pso, '%.2f')], 'FontSize', 12);
text(0.1, 0.3, ['超调量 (°C):', newline, ...
    'Cohen-Coon: ', num2str(overshoot_cc, '%.4f'), newline, ...
    'PSO: ', num2str(overshoot_pso, '%.4f')], 'FontSize', 12);
text(0.5, 0.3, ['稳态误差 (°C):', newline, ...
    'Cohen-Coon: ', num2str(steady_error_cc, '%.4f'), newline, ...
    'PSO: ', num2str(steady_error_pso, '%.4f')], 'FontSize', 12);
axis([0 1 0 1]);
set(gca, 'Visible', 'off');