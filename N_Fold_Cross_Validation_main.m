% Filename:    N_Fold_Cross_Validation_main.m
% Author:      Nicholas Shinn
% Class:       Principles of Data Mining
% Description: Program that performs 10-fold cross-validation using 
%              a decision forest

function N_Fold_Cross_Validation_main(filename)
%HW_09_ngs1112_Mentor - Mentor program for exercise 09 of PDM
%   filename - Name of training file to be used for cross validation

    % Reads in the training data and calculates how many rows would be in
    %each subset for N-Fold cross validation
    training_data = readmatrix(filename);
    
    % Quantizes the data into appropriate bins
    for column = 1:6
        if( column ~= 2 )
            training_data( :, column ) = round(training_data( :, column) / 2) * 2;
        else
            training_data( :, column ) = round(training_data( :, column) / 4) * 4;
        end
    end
    
    % Calculates how many assam and bhutan should be in each subset
    num_rows = size(training_data, 1);
    set_size = num_rows / 10;
    halfset = set_size / 2;
    
    % Splits the assam entries from the bhutan
    assam = training_data( training_data(:, 9) == -1, :);
    bhutan =  training_data( training_data(:, 9) == +1, :);
    
    % Creates two seperate 10-cell arrays to be mixed into training sets
    rowDist = [ halfset halfset halfset halfset halfset halfset halfset halfset halfset halfset ];
    top_half = mat2cell(assam, rowDist);
    bot_half = mat2cell(bhutan, rowDist);
    
    % Mixes a cell from the assam and bhutan arrays into each set,
    % randomizing the entries
    s0 = [ top_half{1}; bot_half{1} ];
    s0 = s0( randperm( size(s0, 1) ), :);
    s1 = [ top_half{2}; bot_half{2} ];
    s1 = s1( randperm( size(s1, 1) ), :);
    s2 = [ top_half{3}; bot_half{3} ];
    s2 = s2( randperm( size(s2, 1) ), :);
    s3 = [ top_half{4}; bot_half{4} ];
    s3 = s3( randperm( size(s3, 1) ), :);
    s4 = [ top_half{5}; bot_half{5} ];
    s4 = s4( randperm( size(s4, 1) ), :);
    s5 = [ top_half{6}; bot_half{6} ];
    s5 = s5( randperm( size(s5, 1) ), :);
    s6 = [ top_half{7}; bot_half{7} ];
    s6 = s6( randperm( size(s6, 1) ), :);
    s7 = [ top_half{8}; bot_half{8} ];
    s7 = s7( randperm( size(s7, 1) ), :);
    s8 = [ top_half{9}; bot_half{9} ];
    s8 = s8( randperm( size(s8, 1) ), :);
    s9 = [ top_half{10}; bot_half{10} ];
    s9 = s9( randperm( size(s9, 1) ), :);
    
    % Combines the cells into a new cell array
    sets = { s0; s1; s2; s3; s4; s5; s6; s7; s8; s9 };
    
    % Possible stump values to test
    N_STUMPS = [ 1 2 4 8 10 20 25 35 50 75 100 150 200 250 300 400 ];
    
    %Sets up trackers
    best_error = Inf;
    best_stumps = Inf;
    best_set = Inf;
    
    % Creates a matrix that will hold data to be graphed later
    graph_data = zeros(16, 2);
    graph_index = 1;
    
    % Iterate through each possible stopping depth
    for stumps = N_STUMPS
        
        % Set up an array to hold individual k-means error
        error_values = zeros(10);
        
        % For each different validation set, create the program and find
        % it's k-means error
        for validation = 1:length(sets)
            error = File_Builder(stumps, validation, sets);
            error_values(validation) = error/1000;
        end
        
        % Once we have the error value for each validation set, find the
        % average error for this depth
        total_error = 0;
        for index = 1:10
            total_error = total_error + error_values(index);
        end
        total_error = total_error/10;
        
        % If this is the current best error, track it
        if( total_error < best_error )
            best_error = total_error;
            best_stumps = stumps;
            best_set = find(min(error_values));
        end
        
        % Add the data to the matrix to be graphed later
        graph_data(graph_index, 1) = stumps;
        graph_data(graph_index, 2) = total_error;
        graph_index = graph_index + 1;
    end

    % Plots the data, marking the point that will be used
    plot( graph_data( :, 1 ), graph_data( :, 2 ) );
    hold on
    plot( best_stumps, best_error, 'r*');
    
    title('Error Versus Different Stopping Depths');
    legend('Error', 'Best Error');
    xlabel('Stopping Depth');
    ylabel('K-Means Error');
    
    % Finally, write the program using the best stopping depth
    File_Builder(best_stumps, best_set, sets);
    
end

function [error] = File_Builder(stumps, validation, sets)

    % Opens the file and creates the header
    if ~exist('out', 'dir')
       mkdir('out');
    end
    answer = fopen('out/N_Fold_Cross_Validation_classifier.m', 'wt');
    addpath(genpath('out'));
    fprintf(answer, 'function N_Fold_Cross_Validation_classifier(filename)\n');
    fprintf(answer, "records = fopen('out/Classifications.csv', 'wt');\n");
    fprintf(answer, 'data = readmatrix(filename);\n');
    fprintf(answer, 'rows = size(data, 1);\n');
    fprintf(answer, 'for row = 1:rows\n');
    fprintf(answer, 'answer = 0;\n');
    
    % Iteratively creates each decision stump
    for N = 1:stumps
                
        % Generates a random feature and threshold from the training sets,
        % excluding the validation set
        possible_sets = setdiff(1:10, validation);
        set_to_use = possible_sets(randi(numel(possible_sets)));
        row = randi( size( sets{set_to_use}, 1 ) );
        feature = randi( size( sets{set_to_use}, 2 ) - 2 );
        threshold = sets{set_to_use}( row, feature );
                
        % Counts how many values from our training data below the threshold
        % are of each respective class
        total_assam = 0;
        total_bhutan = 0;
        for current = 1:length(sets)
            if( current ~= validation )
                lower = sets{current}( sets{current}(:, feature) <= threshold, :);
                total_assam = total_assam + sum( lower(:, 9) == -1);
                total_bhutan = total_bhutan + sum( lower(:, 9) == +1);
            end
        end
                
        % Depending on which class is the majority, writes the stub to vote
        % accordingly
        if( total_assam > total_bhutan )
            fprintf(answer, 'if( data( row, %i ) <= %i )\n', feature, threshold);
            fprintf(answer, "\tanswer = answer - 1;\n");
            fprintf(answer, 'else\n');
            fprintf(answer, "\tanswer = answer + 1;\n");
            fprintf(answer, "end\n\n");
        else
            fprintf(answer, 'if( data( row, %i ) <= %i )\n', feature, threshold);
            fprintf(answer, "\tanswer = answer + 1;\n");
            fprintf(answer, 'else\n');
            fprintf(answer, "\tanswer = answer - 1;\n");
            fprintf(answer, "end\n\n");
        end
                    
    end
    
    % Prints the footer that uses the decisions stumps votes to make the
    % final classification
    fprintf(answer, 'if( answer <= 0 )\n');
    fprintf(answer, "fprintf(records, '-1\\n');\n");
    fprintf(answer, 'else\n');
    fprintf(answer, "fprintf(records, '+1\\n');\n");
    fprintf(answer, "end\n");
    fprintf(answer, 'end\n');
    fprintf(answer, 'fclose(records);\n');
    fprintf(answer, 'end\n');
    fclose(answer);
    
    % Sends the validation set to the new program, pulling the results into
    % a matrix
    writematrix( sets{validation}, "out/CrossValidation.csv");
    N_Fold_Cross_Validation_classifier('out/CrossValidation.csv');
    results = readmatrix('out/Classifications.csv');

    % Calculates and returns the k-means error for this iteration
    error = 0;
    for check_row = 1:size( sets{validation}, 1 )
        error = error + ( sets{validation}(check_row, 9) - results(check_row, 1) )^2;
    end
end

