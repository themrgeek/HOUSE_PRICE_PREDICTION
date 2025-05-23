function [X_train, y_train, X_test, y_test] = load_and_preprocess(filename)
    % 1. Load data with error handling
    try
        data = readtable(filename, 'TextType', 'string');
    catch ME
        error('Failed to load file: %s', ME.message);
    end
    
    fprintf('Raw data loaded: %d rows x %d columns\n', size(data,1), size(data,2));
    
    % 2. Identify numeric columns
    numeric_cols = false(1, width(data));
    for i = 1:width(data)
        col_data = data.(i);
        if isnumeric(col_data) || all(ismissing(col_data) | ~isnan(str2double(string(col_data))))
            numeric_cols(i) = true;
        end
    end
    
    if sum(numeric_cols) < 2
        error('Need at least 2 numeric columns (found %d). Available columns:\n%s', ...
              sum(numeric_cols), strjoin(data.Properties.VariableNames, ', '));
    end
    
    % 3. Convert to numeric matrix
    X = zeros(height(data), sum(numeric_cols)-1);
    y = zeros(height(data), 1);
    
    col_idx = 1;
    for i = find(numeric_cols)
        current_col = data.(i);
        
        if isnumeric(current_col)
            converted = current_col;
        else
            converted = str2double(regexprep(string(current_col), '[^\d\.]', ''));
        end
        
        if i == find(numeric_cols,1,'last') % Last numeric column as target
            y = converted;
        else
            X(:,col_idx) = converted;
            col_idx = col_idx + 1;
        end
    end
    
    % 4. Remove rows with any NaN values
    valid_rows = all(~isnan(X),2) & ~isnan(y);
    X = X(valid_rows,:);
    y = y(valid_rows);
    
    if isempty(y)
        error('All rows removed during cleaning. Check for:\n1. Non-numeric values\n2. Missing data\n3. Invalid numbers');
    end
    
    % 5. Normalize and split
    X = normalize(X, 'range');
    
    rng(42); % For reproducibility
    cv = cvpartition(length(y), 'HoldOut', 0.2);
    
    X_train = X(cv.training,:);
    y_train = y(cv.training);
    X_test = X(cv.test,:);
    y_test = y(cv.test);
    
    fprintf('Final dataset: %d training, %d test samples\n', length(y_train), length(y_test));
end