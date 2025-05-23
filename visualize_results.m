function visualize_results(models, results, X_test, y_test)
    model_names = fieldnames(models);
    
    % 1. Actual vs Predicted Scatter Plots
    figure('Position', [100 100 1200 800]);
    for i = 1:length(model_names)
        name = model_names{i};
        subplot(2,2,i);
        
        scatter(y_test, results.(name).predictions, 30, 'filled');
        hold on;
        plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], 'r--');
        
        title(sprintf('%s (R²=%.2f)', upper(name), results.(name).r2));
        xlabel('Actual Price');
        ylabel('Predicted Price');
        grid on;
    end
    
    % 2. Error Distribution
    figure;
    errors = zeros(length(y_test), length(model_names));
    for i = 1:length(model_names)
        errors(:,i) = y_test - results.(model_names{i}).predictions;
    end
    boxplot(errors, 'Labels', model_names);
    title('Prediction Error Distribution');
    ylabel('Error (Actual - Predicted)');
    grid on;
    
    % 3. Heatmap of Feature Importance
    plot_heatmap(models, X_test, y_test);
end