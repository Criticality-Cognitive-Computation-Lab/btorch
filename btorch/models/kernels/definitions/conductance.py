"""Shared conductance/adaptation stages composed by ALIF and ELIF."""

from .integration import _exp_euler_backward, _exp_euler_forward, _lines


def charge(spec, modes, prepared_values, authoritative_output, *, membrane):
    dt = "dt"
    f = []
    f += _lines("""
                leak_gap = v_prev - E_leak
                leak_term = -g_leak * leak_gap
                potassium_gap = v_prev - E_k
                potassium_term = -g_k_prev * potassium_gap
                membrane_sum = leak_term + potassium_term
            """)
    f += membrane
    f += _lines(f"""
                v_numerator = membrane_sum + x
                v_derivative = v_numerator / c_m
                v_linear = v_linear_numerator / c_m
                {chr(10).join(_exp_euler_forward("v", "v_linear", dt))}
                v_charged = v_prev + v_phi * v_derivative
                g_k_negative = -g_k_prev
                g_k_derivative = g_k_negative / tau_adapt
                g_k_reciprocal = 1.0 / tau_adapt
                g_k_linear = g_k_reciprocal * -1.0
                {chr(10).join(_exp_euler_forward("g_k", "g_k_linear", dt))}
                g_k_decayed = g_k_prev + g_k_phi * g_k_derivative
            """)
    return f


def reverse_charge(spec, modes, prepared_values, authoritative_output, *, voltage):
    dt = "dt"
    b = []
    b += _lines(f"""
                grad_v_derivative = gcharged * v_phi
                grad_v_phi = gcharged * v_derivative
                {chr(10).join(_exp_euler_backward("v", "v_linear", dt))}
                grad_v_numerator = grad_v_derivative / c_m
                grad_v_linear_numerator = grad_v_linear / c_m
                grad_x = grad_v_numerator
                grad_c_m_derivative = -grad_v_derivative * (v_derivative / c_m)
                grad_c_m_linear = -grad_v_linear * (v_linear / c_m)
                grad_p_c_m = grad_c_m_derivative + grad_c_m_linear
                grad_g_leak_derivative = -grad_v_numerator * leak_gap
                grad_p_g_leak = grad_g_leak_derivative - grad_v_linear_numerator
                grad_p_E_leak = grad_v_numerator * g_leak
                grad_g_k_derivative_voltage = -grad_v_numerator * potassium_gap
                grad_g_k_voltage = grad_g_k_derivative_voltage - grad_v_linear_numerator
                grad_p_E_k = grad_v_numerator * g_k_prev
                grad_v_leak = grad_v_numerator * g_leak
                grad_v_potassium = grad_v_numerator * g_k_prev
            """)
    b += voltage
    b += _lines(f"""
                grad_g_k_derivative = grad_s_g_k * g_k_phi
                grad_g_k_phi = grad_s_g_k * g_k_derivative
                {chr(10).join(_exp_euler_backward("g_k", "g_k_linear", dt))}
                grad_g_k_reciprocal = grad_g_k_linear * -1.0
                g_k_reciprocal_square = g_k_reciprocal * g_k_reciprocal
                negative_grad_g_k_reciprocal = -grad_g_k_reciprocal
                grad_tau_linear = negative_grad_g_k_reciprocal * g_k_reciprocal_square
                gk_derivative_over_tau = g_k_derivative / tau_adapt
                neg_grad_gk_derivative = -grad_g_k_derivative
                grad_tau_derivative = neg_grad_gk_derivative * gk_derivative_over_tau
                grad_p_tau_adapt = grad_tau_derivative + grad_tau_linear
                grad_g_k_adapt = grad_s_g_k + (-grad_g_k_derivative / tau_adapt)
                grad_s_g_k = grad_g_k_adapt + grad_g_k_voltage
            """)
    return b


def after_fire(spec, modes, prepared_values, authoritative_output):
    f = []
    f += ["g_k_next = g_k_decayed + dg_k * output"]
    return f


def reverse_reset(spec, modes, prepared_values, authoritative_output):
    reset_gap = "v_charged - v_reset" if spec.hard_reset else "v_threshold - v_reset"
    b = []
    b += ["grad_p_dg_k = grad_s_g_k * output"]
    if not spec.detach_reset:
        b += [
            "gspike = gspike + grad_s_g_k * dg_k",
            f"gspike_reset = (-grad_s_v) * ({reset_gap})",
            "gspike = gspike + gspike_reset",
        ]
    return b
