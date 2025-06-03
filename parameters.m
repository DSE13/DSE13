% define_and_save_params.m

function paramsFilePath = parameters()
    disp('MATLAB: Defining parameters...');

    % --- General Settings ---
    params.vsp_file_path = 'C:\Users\tomsp\OpenVSP-3.43.0-win64\NACA0012_1.vsp3'; % ABSOLUTE PATH
    params.output_data_folder = 'C:\Users\tomsp\OpenVSP-3.43.0-win64'; % Folder for Python to save results

    % --- Analysis Settings ---
    analysis = struct();
    analysis.is_prop_analysis = false; 
    analysis.method_choice = 'vlm'; % 'vlm' or 'panel'
    analysis.ref_span_b = 20.0;    % meters
    analysis.ref_chord_c = 10.0;   % meters
    analysis.xcg = 0;          % meters
    analysis.wake_iterations = int32(3); % Use int32 for Python compatibility if needed
    analysis.symmetry_flag = int32(0); % 0=No, 1=XZ, ...
    analysis.ncpu = int32(feature('numcores'));
    analysis.wake_num_nodes = int32(256);
    analysis.wake_type = int32(1);       % Example: 1 for relaxed
    analysis.far_field_dist_factor = 10.0; 
    params.analysis_settings = analysis;

    % --- Sweep Settings ---
    sweep = struct();
    sweep.sweep_type = 'alpha'; % 'alpha', 'velocity', or 'alpha_beta'
    % For 'alpha' sweep:
    sweep.alpha_sweep_flight_condition_input_type = 'velocity'; % 'velocity' or 'mach'
    sweep.alpha_sweep_flight_velocity_ms = 16; % m/s
    sweep.alpha_npts = int32(5); % Number of Alpha points
    sweep.alpha_start_deg = 0.0;   % Alpha Start (degrees)
    sweep.alpha_end_deg = 8.0;     % Alpha End (degrees)
    if strcmp(sweep.sweep_type,'alpha_beta')
        sweep.beta_npts = 4.0;
        sweep.beta_start_deg = 0.0;
        sweep.beta_end_deg = 3;
    end
    sweep.alpha_sweep_reynolds_mode = 'manual'; % 'calculated' or 'manual'
    if strcmp(sweep.alpha_sweep_reynolds_mode, 'manual')
        sweep.alpha_sweep_manual_reynolds = 1e6;
    end
    params.sweep_settings = sweep;

    % --- Plot Settings (Python will handle plotting or not based on this) ---
    plotting = struct();
    plotting.do_plot = true; % Let Python decide to plot if True
    plotting.x_axis_col = 'Alpha';
    plotting.y_axis_col = 'CL';
    plotting.plot_title_suffix = '(Analysis via JSON)';
    params.plot_settings = plotting;
    
    % --- Save to JSON file ---
    paramsFilePath = fullfile(pwd, 'NACA0012_1_params.json'); % Save in current dir
    try
        jsonData = jsonencode(params, "PrettyPrint", true); % PrettyPrint for readability
        fid = fopen(paramsFilePath, 'w');
        if fid == -1
            error('Cannot create JSON file in %s (check permissions or path)', paramsFilePath);
        end
        fprintf(fid, '%s', jsonData);
        fclose(fid);
        disp(['MATLAB: Parameters saved to JSON: ', paramsFilePath]);
    catch ME
        error('MATLAB: Error saving parameters to JSON: %s', ME.message);
    end
end