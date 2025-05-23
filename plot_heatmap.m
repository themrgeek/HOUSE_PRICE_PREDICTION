function plot_heatmap(models, X_test, y_test)
    model_names = fieldnames(models);
    features = {'List Price', 'Bedrooms', 'Bathrooms', 'Sqft'};
    
    % Calculate permutation importance
    importance = zeros(length(features), length(model_names));
    
    for m = 1:length(model_names)
        name = model_names{m};
        baseline = sqrt(mean((y_test - results.(name).predictions).^2));
        
        for f = 1:size(X_test,2)
            X_perturbed = X_test;
            X_perturbed(:,f) = X_perturbed(randperm(size(X_test,1)),f);
            
            switch name
                case 'ann'
                    pred = predict(models.ann, X_perturbed);
                case {'rf', 'dt'}
                    pred = predict(models.(name), X_perturbed);
                case 'svm'
                    pred = predict(models.svm, X_perturbed);
            end
            
            importance(f,m) = sqrt(mean((y_test - pred).^2)) - baseline;
        end
    end
    
    % Normalize importance
    importance = importance ./ max(importance(:));
    
    % Plot heatmap
    figure;
    h = heatmap(model_names, features, importance);
    h.Title = 'Normalized Feature Importance Across Models';
    h.Colormap = parula;
    h.ColorbarVisible = 'on';
end