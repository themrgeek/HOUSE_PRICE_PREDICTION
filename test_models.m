function results = test_models(models, X_test, y_test)
    results = struct();
    model_names = fieldnames(models);
    
    % Define accuracy thresholds (customize these as needed)
    thresholds = [0.05, 0.10]; % 5% and 10% bounds
    
    for i = 1:length(model_names)
        name = model_names{i};
        current_model = models.(name);
        
        if isempty(current_model)
            continue;
        end
        
        switch name
            case 'ann'
                pred = predict(current_model, X_test);
            case 'lstm'
                % Reshape for LSTM
                X_test_lstm = reshape(X_test', [size(X_test, 2), 1, size(X_test, 1)]);
                pred = predict(current_model, X_test_lstm);
            case {'rf', 'dt'}
                pred = predict(current_model, X_test);
            case 'svm'
                pred = predict(current_model, X_test);
            otherwise
                error('Unknown model type: %s', name);
        end
        
        % Ensure predictions are column vectors
        pred = pred(:);
        y_test = y_test(:);
        
        % Calculate standard metrics
        residuals = y_test - pred;
        results.(name).predictions = pred;
        results.(name).rmse = sqrt(mean(residuals.^2));
        results.(name).mae = mean(abs(residuals));
        results.(name).r2 = 1 - sum(residuals.^2)/sum((y_test - mean(y_test)).^2);
        
        % Calculate percentage accuracy metrics
        relative_errors = abs(residuals)./y_test;
        
        % 1. Percentage within thresholds
        for t = 1:length(thresholds)
            thr = thresholds(t);
            within_thr = mean(relative_errors <= thr) * 100;
            results.(name).(sprintf('accuracy_%dpc', round(thr*100))) = within_thr;
        end
        
        % 2. Median relative accuracy
        results.(name).median_accuracy = (1 - median(relative_errors)) * 100;
    end
    
    % Print results
    fprintf('\nModel Performance:\n');
    fprintf('%-12s %-8s %-8s %-8s %-12s %-12s %-12s\n', ...
            'Model', 'RMSE', 'MAE', 'R²', 'Acc±5%', 'Acc±10%', 'Med Acc%');
    
    for i = 1:length(model_names)
        name = model_names{i};
        if isfield(results, name)
            fprintf('%-12s %-8.2f %-8.2f %-8.2f %-12.1f %-12.1f %-12.1f\n', ...
                name, ...
                results.(name).rmse, ...
                results.(name).mae, ...
                results.(name).r2, ...
                results.(name).accuracy_5pc, ...
                results.(name).accuracy_10pc, ...
                results.(name).median_accuracy);
        end
    end
    
    % Visualize accuracy distributions
    figure;
    hold on;
    colors = lines(length(model_names));
    for i = 1:length(model_names)
        name = model_names{i};
        if isfield(results, name)
            rel_errors = abs(y_test - results.(name).predictions)./y_test;
            [f, x] = ecdf(rel_errors);
            plot(x, f*100, 'Color', colors(i,:), 'LineWidth', 2, 'DisplayName', upper(name));
        end
    end
    xlabel('Relative Error');
    ylabel('Cumulative % of Predictions');
    title('Prediction Accuracy Distribution');
    legend('Location', 'southeast');
    grid on;
    hold off;
end