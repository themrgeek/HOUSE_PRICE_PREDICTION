% MAIN - Real Estate Price Prediction Pipeline
clc; clear; close all;

try
    % 1. Load and preprocess data
    [X_train, y_train, X_test, y_test] = load_and_preprocess('data.xlsx');
    
    % 2. Train models
    models = train_models(X_train, y_train);
    
    % 3. Evaluate models
    results = test_models(models, X_test, y_test);
    
    % 4. Visualize results
    visualize_results(models, results, X_test, y_test);
    
catch ME
    fprintf('Error occurred: %s\n', ME.message);
    fprintf('Line %d in %s\n', ME.stack(1).line, ME.stack(1).name);
end