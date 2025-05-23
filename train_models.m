function models = train_models(X_train, y_train)
    % Initialize models structure
    models = struct();
    
    % Check if data has temporal dimension (for LSTM)
    is_sequential = check_sequential_data(X_train);
    
    % 1. Artificial Neural Network
    layers = [
        featureInputLayer(size(X_train, 2))
        fullyConnectedLayer(64)
        reluLayer()
        fullyConnectedLayer(32)
        reluLayer()
        fullyConnectedLayer(1)
        regressionLayer()
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 32, ...
        'Verbose', 0);
    
    models.ann = trainNetwork(X_train, y_train, layers, options);
    
    % 2. LSTM Network (only if data is sequential)
    if is_sequential
        fprintf('Training LSTM model...\n');
        numFeatures = size(X_train, 2);
        
        lstm_layers = [
            sequenceInputLayer(numFeatures)
            lstmLayer(50, 'OutputMode', 'last')
            fullyConnectedLayer(32)
            reluLayer()
            fullyConnectedLayer(1)
            regressionLayer()
        ];
        
        lstm_options = trainingOptions('adam', ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 16, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', 0);
        
        % Reshape data for LSTM (numFeatures × 1 × numObservations)
        X_train_lstm = reshape(X_train', [size(X_train, 2), 1, size(X_train, 1)]);
        
        models.lstm = trainNetwork(X_train_lstm, y_train, lstm_layers, lstm_options);
    else
        fprintf('Skipping LSTM - input data not sequential\n');
    end
    
    % 3. Random Forest
    models.rf = TreeBagger(50, X_train, y_train, 'Method', 'regression');
    
    % 4. Support Vector Machine
    models.svm = fitrsvm(X_train, y_train, ...
        'KernelFunction', 'gaussian', ...
        'Standardize', true);
    
    % 5. Decision Tree
    models.dt = fitrtree(X_train, y_train);
end

function is_seq = check_sequential_data(X)
    % Simple check for sequential patterns
    % Implement your own logic based on data characteristics
    is_seq = false; % Default to false unless you have time-series data
    
    % Alternative: Check if data has temporal column (like 'date')
    % if any(strcmp(var_names, 'date'))
    %     is_seq = true;
    % end
end