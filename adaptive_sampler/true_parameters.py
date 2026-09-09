"""

True parameters
===============

Value of the parameters used to generate synthetic choices.
Supports P=10, P=20, and P=40 parameter configurations.
"""
SCALE = 1

def get_base_parameters(P, interacted_attributes_only=False):
    """Get base true parameters before applying experimental variations"""

    # Base 10 parameters (same as synthetic_data_generation.ipynb)
    base_params = {
        'beta_rating': 0.75,
        'beta_cost': -0.4,
        'beta_log_dist': -0.60,
        'beta_chinese': 0.83,
        'beta_japanese': 1.25,
        'beta_korean': 0.75,
        'beta_indian': 1,
        'beta_french': 0.95,
        'beta_mexican': 1.30,
        'beta_lebanese': 0.90,
    }
    
    if P >= 20:
        # Add interactions and continuous features (further reduced for balanced peakedness)
        base_params.update({
            'beta_cost_x_logincome': 0.02,    
            'beta_logdist_x_age': -0.02,       
            # 'beta_rating_x_risk': 0.03,       
            'beta_rating_x_edu_sec': 0.15,    
            # 'beta_rating_x_edu_tert': 0.1,   
        })
        
        # 5 continuous restaurant features (further reduced)
        base_params.update({
            'beta_c_16': 0.09,                
            # 'beta_c_17': -0.03,                
            # 'beta_c_18': 0.08,                
            'beta_c_19': -0.04,                
            'beta_c_20': 0.12,   
            'beta_c_21': 0.22,  
            'beta_c_22': -0.18, 
            # 'beta_c_23': 0.19,
            'beta_c_24': 0.28,  
            'beta_c_25': -0.15,             
        })
    
    if P >= 40:
        base_params.update({
            'beta_c_21': 0.22,  'beta_c_22': -0.18, 'beta_c_23': 0.19,
            'beta_c_24': 0.28,  'beta_c_25': -0.15, 'beta_c_26': 0.16,
            'beta_c_27': 0.32,  'beta_c_28': -0.15, 'beta_c_29': 0.24,
            'beta_c_30': 0.26,  'beta_c_31': -0.15, 'beta_c_32': 0.20,
            'beta_c_33': 0.30,  'beta_c_34': -0.15, 'beta_c_35': 0.34,
            'beta_d_36': 0.24,  'beta_d_37': 0.28,  'beta_d_38': -0.22,
            'beta_d_39': 0.20,  'beta_d_40': 0.24,  'beta_d_41': 0.26,
            'beta_d_42': -0.16, 'beta_d_43': 0.22,  'beta_d_44': 0.18,
            'beta_d_45': 0.30,  'beta_d_46': -0.18, 'beta_d_47': 0.16,
            'beta_d_48': -0.20, 'beta_d_49': 0.26,  'beta_d_50': 0.16,
        })

    return base_params

def get_true_parameters(P, interacted_attributes_only=False, M=0):
    """
    Get true parameters for data generation (wrapper for compatibility)
    
    Parameters:
    -----------
    P : int
        Number of parameters (10, 20, or 40)
    interacted_attributes_only : bool
        If True, use only interactions (no c_ or d_ features) for P>=40
    
    Returns:
    --------
    dict : Dictionary of parameter names to values
    """
    params = get_base_parameters(P, interacted_attributes_only=interacted_attributes_only)
    
    # Apply SCALE if needed
    if M == 1:
        SCALE =0.4
        params = {k: v * SCALE for k, v in params.items()}
    
    return params

# Default to P=10 for backward compatibility
# true_parameters = get_true_parameters(10)
