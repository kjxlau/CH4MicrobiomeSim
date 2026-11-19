import os
import sys

# 1. Disable Gradio SSR proxy to prevent container crashes
os.environ["GRADIO_SSR_MODE"] = "false"

# 2. Patch asyncio event loop destructor on Python 3.12+ (Fixes: "ValueError: Invalid file descriptor: -1")
import asyncio.base_events as _base_events
_orig_del = getattr(_base_events.BaseEventLoop, "__del__", None)
def _safe_event_loop_del(self):
    try:
        if _orig_del:
            _orig_del(self)
    except ValueError as e:
        if "Invalid file descriptor" not in str(e):
            raise
_base_events.BaseEventLoop.__del__ = _safe_event_loop_del

import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import random
import psutil

# ZeroGPU compatibility for Hugging Face Spaces
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

# =========================================================================
# Robust psutil.Process patch for Container environments (PID 0)
# =========================================================================
_original_Process = psutil.Process

class SafeDummyProcess:
    def name(self): return "docker_init"
    def is_running(self): return True
    def status(self): return psutil.STATUS_RUNNING
    def cmdline(self): return ["docker_init"]
    def ppid(self): return 0
    def parent(self): return None
    def children(self, recursive=False): return []
    def terminate(self): pass
    def kill(self): pass
    def wait(self, timeout=None): pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

def _patched_Process(pid=None):
    if pid == 0:
        return SafeDummyProcess()
    try:
        return _original_Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return SafeDummyProcess()

psutil.Process = _patched_Process
# =========================================================================

import pm4py
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.utils import petri_utils

if not os.path.exists("temp_images"):
    os.makedirs("temp_images")

TOTAL_SIMULATION_DAYS = 730

# =========================================================================
# LONGITUDINAL EMPIRICAL METAGENOMIC DATA (Weeks 3, 6, and 10)
# =========================================================================
WEEKLY_CORR_DATA = {
    3: {
        "CH4": {
            "Methanothrix":     {"r": 0.617138, "p_val": 8.13e-6,   "p_adj": 0.003732, "sig": "***"},
            "Methanocella":     {"r": 0.530168, "p_val": 0.000214, "p_adj": 0.021492, "sig": "***"},
            "Methanobacterium": {"r": 0.513912, "p_val": 0.000360, "p_adj": 0.032038, "sig": "***"},
            "Desulfomonile":    {"r": 0.500907, "p_val": 0.000534, "p_adj": 0.038908, "sig": "***"},
            "Methanoregula":    {"r": 0.500788, "p_val": 0.000536, "p_adj": 0.038908, "sig": "***"},
            "Methanosarcina":   {"r": 0.367018, "p_val": 0.014261, "p_adj": 0.314958, "sig": "*"},
            "Methylocaldum":    {"r": 0.341478, "p_val": 0.023294, "p_adj": 0.368480, "sig": "*"},
            "Sulfuricaulis":    {"r": 0.274771, "p_val": 0.071063, "p_adj": 0.493586, "sig": "ns"},
            "Syntrophus":       {"r": 0.274198, "p_val": 0.071678, "p_adj": 0.493586, "sig": "ns"},
            "Nitrospira":       {"r": -0.390378, "p_val": 0.008798, "p_adj": 0.256595, "sig": "**"}
        },
        "N2O": {
            "Nitrospira":       {"r": 0.357401, "p_val": 0.017231, "p_adj": 0.345483, "sig": "*"},
            "Methanothrix":     {"r": -0.300661, "p_val": 0.047357, "p_adj": 0.426745, "sig": "*"},
            "Methanocella":     {"r": -0.336807, "p_val": 0.025380, "p_adj": 0.368480, "sig": "*"},
            "Methylocaldum":    {"r": -0.333442, "p_val": 0.026977, "p_adj": 0.376266, "sig": "*"},
            "Desulfomonile":    {"r": -0.207555, "p_val": 0.176406, "p_adj": 0.694371, "sig": "ns"},
            "Methanobacterium": {"r": -0.199343, "p_val": 0.194531, "p_adj": 0.710770, "sig": "ns"},
            "Syntrophus":       {"r": -0.185922, "p_val": 0.226919, "p_adj": 0.743884, "sig": "ns"},
            "Methanoregula":    {"r": -0.077886, "p_val": 0.615295, "p_adj": 0.890055, "sig": "ns"},
            "Methanosarcina":   {"r": -0.068207, "p_val": 0.659993, "p_adj": 0.908549, "sig": "ns"},
            "Sulfuricaulis":    {"r": -0.025072, "p_val": 0.871662, "p_adj": 0.954777, "sig": "ns"}
        }
    },
    6: {
        "CH4": {
            "Methanoregula":    {"r": 0.620846, "p_val": 2.68e-8,   "p_adj": 4.57e-5,  "sig": "***"},
            "Syntrophus":       {"r": 0.561556, "p_val": 9.32e-7,   "p_adj": 0.000398, "sig": "***"},
            "Methanothrix":     {"r": 0.554867, "p_val": 1.33e-6,   "p_adj": 0.000419, "sig": "***"},
            "Desulfomonile":    {"r": 0.442411, "p_val": 0.000200, "p_adj": 0.017035, "sig": "***"},
            "Methanocella":     {"r": 0.405313, "p_val": 0.000735, "p_adj": 0.044811, "sig": "***"},
            "Methylocaldum":    {"r": 0.345163, "p_val": 0.004535, "p_adj": 0.140668, "sig": "**"},
            "Methanosarcina":   {"r": 0.235345, "p_val": 0.057140, "p_adj": 0.569011, "sig": "ns"},
            "Methanobacterium": {"r": 0.179571, "p_val": 0.149098, "p_adj": 0.698419, "sig": "ns"},
            "Sulfuricaulis":    {"r": 0.073858, "p_val": 0.555615, "p_adj": 0.884216, "sig": "ns"},
            "Nitrospira":       {"r": -0.243752, "p_val": 0.048578, "p_adj": 0.540755, "sig": "*"}
        },
        "N2O": {
            "Methanocella":     {"r": 0.207667, "p_val": 0.094297, "p_adj": 0.637435, "sig": "ns"},
            "Methylocaldum":    {"r": 0.131950, "p_val": 0.290917, "p_adj": 0.800492, "sig": "ns"},
            "Desulfomonile":    {"r": 0.109310, "p_val": 0.382286, "p_adj": 0.823044, "sig": "ns"},
            "Methanobacterium": {"r": 0.075734, "p_val": 0.545590, "p_adj": 0.884216, "sig": "ns"},
            "Methanosarcina":   {"r": 0.073983, "p_val": 0.554946, "p_adj": 0.884216, "sig": "ns"},
            "Sulfuricaulis":    {"r": -0.019306, "p_val": 0.877721, "p_adj": 0.993166, "sig": "ns"},
            "Methanothrix":     {"r": -0.029419, "p_val": 0.814607, "p_adj": 0.982134, "sig": "ns"},
            "Nitrospira":       {"r": -0.080508, "p_val": 0.520487, "p_adj": 0.881562, "sig": "ns"},
            "Methanoregula":    {"r": -0.097594, "p_val": 0.435645, "p_adj": 0.859835, "sig": "ns"},
            "Syntrophus":       {"r": -0.136677, "p_val": 0.273824, "p_adj": 0.799476, "sig": "ns"}
        }
    },
    10: {
        "CH4": {
            "Methylocaldum":    {"r": 0.208143, "p_val": 0.101633, "p_adj": 0.603226, "sig": "ns"},
            "Methanobacterium": {"r": 0.071715, "p_val": 0.576477, "p_adj": 0.908947, "sig": "ns"},
            "Desulfomonile":    {"r": 0.035418, "p_val": 0.782869, "p_adj": 0.956694, "sig": "ns"},
            "Nitrospira":       {"r": -0.034533, "p_val": 0.788172, "p_adj": 0.956913, "sig": "ns"},
            "Syntrophus":       {"r": -0.034838, "p_val": 0.786340, "p_adj": 0.956694, "sig": "ns"},
            "Sulfuricaulis":    {"r": -0.050562, "p_val": 0.693923, "p_adj": 0.934971, "sig": "ns"},
            "Methanosarcina":   {"r": -0.044374, "p_val": 0.729852, "p_adj": 0.947190, "sig": "ns"},
            "Methanothrix":     {"r": -0.103036, "p_val": 0.421637, "p_adj": 0.876536, "sig": "ns"},
            "Methanocella":     {"r": -0.168125, "p_val": 0.187796, "p_adj": 0.741261, "sig": "ns"},
            "Methanoregula":    {"r": -0.237775, "p_val": 0.060587, "p_adj": 0.478565, "sig": "ns"}
        },
        "N2O": {
            "Nitrospira":       {"r": 0.222101, "p_val": 0.080206, "p_adj": 0.562161, "sig": "ns"},
            "Methanoregula":    {"r": 0.110837, "p_val": 0.387151, "p_adj": 0.863866, "sig": "ns"},
            "Syntrophus":       {"r": 0.000675, "p_val": 0.995811, "p_adj": 0.996923, "sig": "ns"},
            "Desulfomonile":    {"r": -0.062678, "p_val": 0.625542, "p_adj": 0.917794, "sig": "ns"},
            "Methanothrix":     {"r": -0.086723, "p_val": 0.499148, "p_adj": 0.902104, "sig": "ns"},
            "Sulfuricaulis":    {"r": -0.162793, "p_val": 0.202387, "p_adj": 0.760332, "sig": "ns"},
            "Methanocella":     {"r": -0.291672, "p_val": 0.020373, "p_adj": 0.268451, "sig": "*"},
            "Methanosarcina":   {"r": -0.296503, "p_val": 0.018294, "p_adj": 0.249024, "sig": "*"},
            "Methylocaldum":    {"r": -0.347061, "p_val": 0.005324, "p_adj": 0.127017, "sig": "**"},
            "Methanobacterium": {"r": -0.371442, "p_val": 0.002725, "p_adj": 0.084179, "sig": "**"}
        }
    }
}

# ==========================================
# 1. CH4MOD MODEL FUNCTIONS (Macro)
# ==========================================
K1 = 0.027; K2 = 0.003; K3 = 0.000008
EH0 = -150.0

def compute_ti(t_soil):
    ti = 3.0 ** ((t_soil - 30.0) / 10.0)
    return min(ti, 1.0)

def compute_eh_index(eh_val, wd):
    if wd > -5.0:
        eh_val = eh_val - 0.13 * (eh_val + 150.0)
    else:
        eh_val = eh_val - 0.025 * (eh_val - 150.0)
    
    if eh_val < -150.0:
        eh_index = 1.0
    else:
        eh_index = np.exp(-1.7 * ((150.0 + eh_val) / 150.0))
    return eh_index, eh_val

def compute_w_grass_multi(w_arr, w_max, w0, seasons, t_air_arr, r, c):
    tem = 25.0
    sdd = 0.0
    current_season_idx = -1
    
    for day in range(len(w_arr)):
        day_num = day + 1 
        in_season = False
        
        for idx, (tgd, tgmax, tgend) in enumerate(seasons):
            if tgd <= day_num <= tgend:
                in_season = True
                if current_season_idx != idx:
                    current_season_idx = idx
                    sdd = 0.0 
                    
                if day_num <= tgmax:
                    val = w_max / (1.0 + (w_max - w0) / w0 * np.exp(-r * (day_num - tgd)))
                    w_arr[day] = val
                else:
                    if t_air_arr[day] < tem: 
                        sdd += (tem - t_air_arr[day])
                    prev_w = w_arr[day-1] if day > 0 else w0
                    w_arr[day] = max(0.0, prev_w - c * sdd)
                break
                
        if not in_season:
            w_arr[day] = 0.0
            current_season_idx = -1

def fcp(t_index, vi, w_val):
    return 1.8 * 0.001 * vi * t_index * (w_val ** 1.25)

def compute_cl_shoot_ana(l_val, k1, k2, t_index, wd, f_shoot_n, f_shoot_s):
    if wd > 50.0: omega = 0.005
    elif 0.0 < wd <= 50.0: omega = 0.0345
    else: omega = 0.0

    ana = l_val * wd * omega * t_index * 0.3 * (f_shoot_n * k1 + f_shoot_s * k2)
    aer = l_val * (1.0 - wd * omega) * t_index * 0.12 * (f_shoot_n * k1 + f_shoot_s * k2)
    l_val = l_val - max(0, ana) - max(0, aer)
    return max(0, ana), l_val

def compute_cl_root_ana(l_root, k1, k2, t_index, wd, f_root_n, f_root_s):
    term = t_index * (f_root_n * k1 + f_root_s * k2)
    if wd > -5.0:
        ana = l_root * term * 0.3
        aer = 0.0
    else:
        ana = l_root * ((160.0 + wd) / 160.0) * term * 0.3
        aer = l_root * (-wd / 160.0) * term * 0.12
    l_root = l_root - ana - aer
    return ana, l_root

def compute_cl_soil(ds, k3, t_index, wd, h):
    if wd >= 0.0:
        cl = ds * k3 * t_index * 0.3 / 0.58
    elif -100.0 < wd < 0.0:
        cl = (ds / 0.58) * k3 * t_index * 0.3 * ((h + wd) / h)
    else:
        cl = 0.0
    ds -= cl
    return cl, ds

def compute_soil_index(sand_fraction):
    return 0.325 + 0.0225 * sand_fraction * 100.0    

def compute_cl_om(om_n, om_s, k1, k2, t_index, soil_index):
    total_om = om_n + om_s
    if total_om <= 0:
        return 0.0, 0.0, 0.0

    decomp = 0.65 * soil_index * t_index * (k1 * om_n + k2 * om_s)
    frac_n = om_n / total_om
    frac_s = om_s / total_om
    
    actual_decomp_n = min(om_n, decomp * frac_n)
    actual_decomp_s = min(om_s, decomp * frac_s)
    
    om_n -= actual_decomp_n
    om_s -= actual_decomp_s
    total_decomp = actual_decomp_n + actual_decomp_s
    return total_decomp, om_n, om_s

def run_ch4mod(
    tsoil_file, wd_file, tair_file, 
    wmax, dsom, vi, r_growth,
    omn_input, oms_input, sand_fraction,
    s1_enable, s1_tgd, s1_tgmax, s1_tgend,
    s2_enable, s2_tgd, s2_tgmax, s2_tgend,
    s3_enable, s3_tgd, s3_tgmax, s3_tgend,
    wd1_start, wd1_end, wd1_val,
    wd2_start, wd2_end, wd2_val,
    wd3_start, wd3_end, wd3_val,
    tair_offset, wd_offset,
    inj1_enable, inj1_day, inj1_omn, inj1_oms,
    inj2_enable, inj2_day, inj2_omn, inj2_oms,
    inj3_enable, inj3_day, inj3_omn, inj3_oms,
    *args, **kwargs
):
    plt.close('all')
    seasons = []
    if s1_enable: seasons.append((int(s1_tgd), int(s1_tgmax), int(s1_tgend)))
    if s2_enable: seasons.append((int(s2_tgd), int(s2_tgmax), int(s2_tgend)))
    if s3_enable: seasons.append((int(s3_tgd), int(s3_tgmax), int(s3_tgend)))

    DEFAULT_DAYS = TOTAL_SIMULATION_DAYS 

    try:
        if tsoil_file is not None: 
            tsoil = np.loadtxt(tsoil_file.name)
        else: 
            tsoil = 25.0 + 5.0 * np.sin(2.0 * np.pi * np.arange(DEFAULT_DAYS) / 365.0)
        
        if wd_file is not None: 
            wd = np.loadtxt(wd_file.name)
        else: 
            wd = np.zeros(DEFAULT_DAYS)
            for d in range(DEFAULT_DAYS):
                day_num = d + 1
                if wd1_start <= day_num <= wd1_end: wd[d] = wd1_val
                elif wd2_start <= day_num <= wd2_end: wd[d] = wd2_val
                elif wd3_start <= day_num <= wd3_end: wd[d] = wd3_val
            
        if tair_file is not None: 
            tair_data = np.loadtxt(tair_file.name)
        else: 
            tair_data = tsoil - 2.0
            
    except Exception as e:
        raise gr.Error(f"Error reading input files: {str(e)}")

    DAYS = min(len(tsoil), len(wd), len(tair_data))
    tsoil, wd, tair_data = tsoil[:DAYS], wd[:DAYS], tair_data[:DAYS]
    
    tsoil = tsoil + tair_offset
    tair_data = tair_data + tair_offset
    wd = wd + wd_offset
    
    W0 = 50.0; POX = 0.2; TVEG = 1.0; FR = 0.1
    FSN = 0.04; FSS = 0.04; FRN = 0.04; FRS = 0.04
    
    w_arr = np.zeros(DAYS)
    compute_w_grass_multi(w_arr, wmax, W0, seasons, tair_data, r_growth, 0.05)

    l_pool = wmax * 0.95
    l_root_day = 0.0
    eh_val = EH0
    om_n_pool = omn_input
    om_s_pool = oms_input

    results = {"Day":[], "Biomass":[], "T_Soil":[], "Water":[], "Prod_CH4":[], "Emission_Bubble":[], "Emission_Total":[], "Eh_Redox":[]}
    
    for d in range(DAYS):
        day_num = d + 1
        
        if inj1_enable and day_num == inj1_day:
            om_n_pool += inj1_omn
            om_s_pool += inj1_oms
        if inj2_enable and day_num == inj2_day:
            om_n_pool += inj2_omn
            om_s_pool += inj2_oms
        if inj3_enable and day_num == inj3_day:
            om_n_pool += inj3_omn
            om_s_pool += inj3_oms
        
        ti = compute_ti(tsoil[d])
        eh_idx, eh_val = compute_eh_index(eh_val, wd[d])
        w_day = w_arr[d]
        
        cp = 0.0; cl_shoot = 0.0; cl_root = 0.0
        in_season = False

        for (tgd, tgmax, tgend) in seasons:
            if tgd <= day_num <= tgend:
                in_season = True
                tgu = tgd + 60
                if day_num <= tgu:
                    cp = fcp(ti, vi, w_day)
                    delta_l_root = (l_pool * FR) / (tgu - tgd)
                    l_root_day += delta_l_root
                    cl_root, l_root_day = compute_cl_root_ana(l_root_day, K1, K2, ti, wd[d], FRN, FRS)
                    cl_shoot, l_pool = compute_cl_shoot_ana(l_pool, K1, K2, ti, wd[d], FSN, FSS)
                elif day_num <= tgmax:
                    cp = fcp(ti, vi, w_day)
                    cl_root, l_root_day = compute_cl_root_ana(l_root_day, K1, K2, ti, wd[d], FRN, FRS)
                    cl_shoot, l_pool = compute_cl_shoot_ana(l_pool, K1, K2, ti, wd[d], FSN, FSS)
                else:
                    cp = fcp(ti, vi, w_day)
                    prev_w = w_arr[d-1] if d > 0 else W0
                    l_pool += max(0, prev_w - w_day)
                    cl_shoot, l_pool = compute_cl_shoot_ana(l_pool, K1, K2, ti, wd[d], FSN, FSS)
                    cl_root, l_root_day = compute_cl_root_ana(l_root_day, K1, K2, ti, wd[d], FRN, FRS)
                break
                
        if not in_season:
            cp = 0.0
            if d > 0:
                prev_w = w_arr[d-1]
                if prev_w > 0 and w_day == 0:
                    l_pool += prev_w
            cl_shoot, l_pool = compute_cl_shoot_ana(l_pool, K1, K2, ti, wd[d], FSN, FSS)
            cl_root, l_root_day = compute_cl_root_ana(l_root_day, K1, K2, ti, wd[d], FRN, FRS)

        SOIL_INDEX = compute_soil_index(sand_fraction)
        cl_om_val, om_n_pool, om_s_pool = compute_cl_om(om_n_pool, om_s_pool, K1, K2, ti, SOIL_INDEX)
        cl_soil_val, dsom = compute_cl_soil(dsom, K3, ti, wd[d], 100.0)
        
        production = 0.27 * (eh_idx * (cp + cl_shoot) + cl_root + cl_soil_val + cl_om_val)
        
        e_bub = 0.0
        if wd[d] > 0:
            e_bub = (production - 0.002) * 0.4 * (1.0 - w_day / max(1, wmax))
        
        e_diff = max(0, (production - 0.06) * 0.3 * 0.1)
        e_plant = (production - max(0, e_bub)) * (1.0 - POX) * TVEG
        total_emission = max(0, e_plant) + max(0, e_bub) + e_diff
        conv = 1000.0 / 24.0
        
        results["Day"].append(day_num)
        results["Biomass"].append(round(w_day, 2))
        results["T_Soil"].append(round(tsoil[d], 2))
        results["Water"].append(round(wd[d], 2))
        results["Prod_CH4"].append(round(production * conv, 3))
        results["Emission_Bubble"].append(round(max(0, e_bub) * conv, 3)) 
        results["Emission_Total"].append(round(total_emission * conv, 3))
        results["Eh_Redox"].append(round(eh_val, 2))

    df = pd.DataFrame(results)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.plot(df["Day"], df["Biomass"], color="green", linewidth=2)
    ax1.set_ylabel("Biomass (g/m²)", fontweight="bold")
    ax1.set_title("Crop Biomass Across 2-Year Horizon (730 Days)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(df["Day"], df["Emission_Total"], color="red", linewidth=2)
    ax2.set_ylabel("Total CH4 (mg/m²/h)", fontweight="bold")
    ax2.set_title("Daily Methane Emissions", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    ax3.plot(df["Day"], df["Water"], color="blue", linewidth=2)
    ax3.set_xlabel("Simulation Day (1 to 730)", fontweight="bold")
    ax3.set_ylabel("Water Depth (cm)", fontweight="bold")
    ax3.set_title("Flooding & Drainage Status (WD)", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    
    for ax in (ax1, ax2, ax3):
        ax.axvline(x=365, color="gray", linestyle=":", alpha=0.8, label="Year 1/2 Cutoff" if ax == ax1 else "")
    ax1.legend(loc="upper right")

    plt.tight_layout()
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, "Simulation_Results.csv")
    df.to_csv(out_path, index=False)

    return fig, df, out_path

# =========================================================================
# 2. PETRI NET MICROBIAL FUNCTIONS (CALIBRATED)
# =========================================================================
Y_DESULFOMONILE = 1.00
Y_SULFURICAULIS = 1.15
Y_ACETATE = 1.20
Y_NITROSPIRA = 1.10
Y_METHYLOC = 1.15

def create_ecosystem_net(tillerno_early, tillerno_mid, tillerno_late):
    net = pm4py.objects.petri_net.obj.PetriNet("Empirical Syntrophic Microbiome Net")

    CH4_in_soil = pm4py.objects.petri_net.obj.PetriNet.Place("CH4 in soil")
    CH4_in_air = pm4py.objects.petri_net.obj.PetriNet.Place("CH4 in air")
    
    methanothrix = pm4py.objects.petri_net.obj.PetriNet.Place("Methanothrix")
    methanobacterium = pm4py.objects.petri_net.obj.PetriNet.Place("Methanobacterium")
    methanosarcina = pm4py.objects.petri_net.obj.PetriNet.Place("Methanosarcina")
    methanocella = pm4py.objects.petri_net.obj.PetriNet.Place("Methanocella")
    methanoregula = pm4py.objects.petri_net.obj.PetriNet.Place("Methanoregula")
    
    desulfomonile = pm4py.objects.petri_net.obj.PetriNet.Place("Desulfomonile")
    syntrophus = pm4py.objects.petri_net.obj.PetriNet.Place("Syntrophus")
    sulfuricaulis = pm4py.objects.petri_net.obj.PetriNet.Place("Sulfuricaulis")
    nitrospira = pm4py.objects.petri_net.obj.PetriNet.Place("Nitrospira")
    methylocaldum = pm4py.objects.petri_net.obj.PetriNet.Place("Methylocaldum")
    
    aerobic = pm4py.objects.petri_net.obj.PetriNet.Place("Aerobic")
    acetate = pm4py.objects.petri_net.obj.PetriNet.Place("Acetate")
    anaerobic = pm4py.objects.petri_net.obj.PetriNet.Place("Anaerobic")
    substrates = pm4py.objects.petri_net.obj.PetriNet.Place("Substrates")
    organic_matter = pm4py.objects.petri_net.obj.PetriNet.Place("Organic_Matter")
    drip = pm4py.objects.petri_net.obj.PetriNet.Place("Drip")
    flood = pm4py.objects.petri_net.obj.PetriNet.Place("Flood")
    nitrate = pm4py.objects.petri_net.obj.PetriNet.Place("Nitrate")
    n2o = pm4py.objects.petri_net.obj.PetriNet.Place("N2O")
    sulphate = pm4py.objects.petri_net.obj.PetriNet.Place("Sulphate")
    h2s = pm4py.objects.petri_net.obj.PetriNet.Place("H2S")
    n_fertilizer = pm4py.objects.petri_net.obj.PetriNet.Place("N Fertilizer")
    days21 = pm4py.objects.petri_net.obj.PetriNet.Place("21days")
    days60 = pm4py.objects.petri_net.obj.PetriNet.Place("60days")
    days90 = pm4py.objects.petri_net.obj.PetriNet.Place("90days")
    co2 = pm4py.objects.petri_net.obj.PetriNet.Place("CO2")
    
    net.places.update([
        CH4_in_soil, CH4_in_air, methanothrix, methanobacterium, methanosarcina,
        methanocella, methanoregula, desulfomonile, syntrophus, sulfuricaulis,
        nitrospira, methylocaldum, aerobic, anaerobic, substrates, organic_matter,
        drip, flood, nitrate, n2o, acetate, sulphate, h2s, n_fertilizer,
        days21, days60, days90, co2
    ])

    mg_thrix  = pm4py.objects.petri_net.obj.PetriNet.Transition("Methanothrix_Acetoclastic_MG")
    mg_bact   = pm4py.objects.petri_net.obj.PetriNet.Transition("Methanobacterium_Hydrogeno_MG")
    mg_sarc   = pm4py.objects.petri_net.obj.PetriNet.Transition("Methanosarcina_Mixotrophic_MG")
    mg_cella  = pm4py.objects.petri_net.obj.PetriNet.Transition("Methanocella_Hydrogeno_MG")
    mg_regula = pm4py.objects.petri_net.obj.PetriNet.Transition("Methanoregula_Hydrogeno_MG")
    
    syntroph_ferm   = pm4py.objects.petri_net.obj.PetriNet.Transition("Syntrophus_Fermentation")
    desulf_red      = pm4py.objects.petri_net.obj.PetriNet.Transition("Desulfomonile_Sulfate_Reduction")
    sulfuri_ox      = pm4py.objects.petri_net.obj.PetriNet.Transition("Sulfuricaulis_Sulfur_Oxidation")
    sulfuri_ox_oxic = pm4py.objects.petri_net.obj.PetriNet.Transition("Sulfuricaulis_Oxic_Growth")
    
    O2_toxicity = pm4py.objects.petri_net.obj.PetriNet.Transition("O2_Toxicity")
    acetogenesis = pm4py.objects.petri_net.obj.PetriNet.Transition("Acetogenesis")
    decomposition = pm4py.objects.petri_net.obj.PetriNet.Transition("Decomposition")
    flood_event = pm4py.objects.petri_net.obj.PetriNet.Transition("Flood_Event")
    anaerobic_decay = pm4py.objects.petri_net.obj.PetriNet.Transition("Anaerobic_Decay")
    O2_consumption = pm4py.objects.petri_net.obj.PetriNet.Transition("O2_Consumption")
    drain_event = pm4py.objects.petri_net.obj.PetriNet.Transition("Drain_Event")
    nitrification = pm4py.objects.petri_net.obj.PetriNet.Transition("Nitrospira_Nitrification")
    denitrification = pm4py.objects.petri_net.obj.PetriNet.Transition("Denitrification")
    tillering = pm4py.objects.petri_net.obj.PetriNet.Transition("Tillering")
    heading = pm4py.objects.petri_net.obj.PetriNet.Transition("Heading")
    ripening = pm4py.objects.petri_net.obj.PetriNet.Transition("Ripening")
    methanotrophy = pm4py.objects.petri_net.obj.PetriNet.Transition("Methylocaldum_Methanotrophy")
    aerobic_oxidation = pm4py.objects.petri_net.obj.PetriNet.Transition("Aerobic_Oxidation")
    
    net.transitions.update([
        mg_thrix, mg_bact, mg_sarc, mg_cella, mg_regula,
        syntroph_ferm, desulf_red, sulfuri_ox, sulfuri_ox_oxic,
        O2_toxicity, acetogenesis, decomposition, flood_event, anaerobic_decay,
        O2_consumption, drain_event, nitrification, denitrification,
        tillering, heading, ripening, methanotrophy, aerobic_oxidation
    ])
    
    # 1. Nitrospira Nitrification (active in Drip and Flood oxic microzones)
    petri_utils.add_arc_from_to(n_fertilizer, nitrification, net, weight=1)
    petri_utils.add_arc_from_to(nitrospira, nitrification, net, weight=2)
    petri_utils.add_arc_from_to(aerobic, nitrification, net, weight=2)
    petri_utils.add_arc_from_to(nitrification, nitrospira, net, weight=6)
    petri_utils.add_arc_from_to(nitrification, nitrate, net, weight=2)
    
    petri_utils.add_arc_from_to(nitrate, denitrification, net, weight=1)
    petri_utils.add_arc_from_to(substrates, denitrification, net, weight=2)
    petri_utils.add_arc_from_to(denitrification, n2o, net, weight=1)
    petri_utils.add_arc_from_to(denitrification, co2, net, weight=1)

    # 2. Syntrophus Fermentation
    petri_utils.add_arc_from_to(syntrophus, syntroph_ferm, net, weight=2)
    petri_utils.add_arc_from_to(substrates, syntroph_ferm, net, weight=2)
    petri_utils.add_arc_from_to(syntroph_ferm, syntrophus, net, weight=3)
    petri_utils.add_arc_from_to(syntroph_ferm, acetate, net, weight=5)
    petri_utils.add_arc_from_to(syntroph_ferm, co2, net, weight=1)

    # 3. Methanogenesis Pathways (balanced consumption vs output)
    # Methanothrix (minor in Flood ~1-2%, 0 in Drip)
    petri_utils.add_arc_from_to(methanothrix, mg_thrix, net, weight=2)
    petri_utils.add_arc_from_to(acetate, mg_thrix, net, weight=2)
    petri_utils.add_arc_from_to(mg_thrix, methanothrix, net, weight=5)
    petri_utils.add_arc_from_to(mg_thrix, CH4_in_soil, net, weight=2)
    petri_utils.add_arc_from_to(mg_thrix, co2, net, weight=1)

    # Methanobacterium (moderate in Flood ~3-7%)
    petri_utils.add_arc_from_to(methanobacterium, mg_bact, net, weight=2)
    petri_utils.add_arc_from_to(substrates, mg_bact, net, weight=2)
    petri_utils.add_arc_from_to(mg_bact, methanobacterium, net, weight=4)
    petri_utils.add_arc_from_to(mg_bact, CH4_in_soil, net, weight=2)

    # Methanosarcina (main methanogen ~7-16% in Flood, ~2-4% in Drip)
    petri_utils.add_arc_from_to(methanosarcina, mg_sarc, net, weight=3)
    petri_utils.add_arc_from_to(acetate, mg_sarc, net, weight=2)
    petri_utils.add_arc_from_to(mg_sarc, methanosarcina, net, weight=6)
    petri_utils.add_arc_from_to(mg_sarc, CH4_in_soil, net, weight=2)
    petri_utils.add_arc_from_to(mg_sarc, co2, net, weight=1)

    # Methanocella (moderate in Flood ~3-6%)
    petri_utils.add_arc_from_to(methanocella, mg_cella, net, weight=2)
    petri_utils.add_arc_from_to(substrates, mg_cella, net, weight=2)
    petri_utils.add_arc_from_to(mg_cella, methanocella, net, weight=4)
    petri_utils.add_arc_from_to(mg_cella, CH4_in_soil, net, weight=2)

    # Methanoregula (minor in Flood ~1-3%)
    petri_utils.add_arc_from_to(methanoregula, mg_regula, net, weight=2)
    petri_utils.add_arc_from_to(substrates, mg_regula, net, weight=2)
    petri_utils.add_arc_from_to(mg_regula, methanoregula, net, weight=3)
    petri_utils.add_arc_from_to(mg_regula, CH4_in_soil, net, weight=2)

    # 4. Desulfomonile Sulfate Reduction
    petri_utils.add_arc_from_to(desulfomonile, desulf_red, net, weight=2)
    petri_utils.add_arc_from_to(acetate, desulf_red, net, weight=2)
    petri_utils.add_arc_from_to(sulphate, desulf_red, net, weight=1)
    petri_utils.add_arc_from_to(desulf_red, desulfomonile, net, weight=3)
    petri_utils.add_arc_from_to(desulf_red, h2s, net, weight=2)
    petri_utils.add_arc_from_to(desulf_red, co2, net, weight=2)

    # 5. Sulfuricaulis Sulfur Oxidation (Flood anaerobic route + Drip oxic route)
    petri_utils.add_arc_from_to(sulfuricaulis, sulfuri_ox, net, weight=2)
    petri_utils.add_arc_from_to(h2s, sulfuri_ox, net, weight=1)
    petri_utils.add_arc_from_to(nitrate, sulfuri_ox, net, weight=1)
    petri_utils.add_arc_from_to(sulfuri_ox, sulfuricaulis, net, weight=4)
    petri_utils.add_arc_from_to(sulfuri_ox, sulphate, net, weight=1)

    petri_utils.add_arc_from_to(sulfuricaulis, sulfuri_ox_oxic, net, weight=2)
    petri_utils.add_arc_from_to(aerobic, sulfuri_ox_oxic, net, weight=2)
    petri_utils.add_arc_from_to(substrates, sulfuri_ox_oxic, net, weight=1)
    petri_utils.add_arc_from_to(sulfuri_ox_oxic, sulfuricaulis, net, weight=4)
    petri_utils.add_arc_from_to(sulfuri_ox_oxic, sulphate, net, weight=1)

    # 6. Fermentation & Decomposition
    petri_utils.add_arc_from_to(organic_matter, decomposition, net, weight=1)
    petri_utils.add_arc_from_to(decomposition, substrates, net, weight=20)
    petri_utils.add_arc_from_to(substrates, acetogenesis, net, weight=15)
    petri_utils.add_arc_from_to(anaerobic, acetogenesis, net, weight=10)
    petri_utils.add_arc_from_to(acetogenesis, acetate, net, weight=80)

    # 7. Redox Controls (Nitrospira is NOT consumed by anaerobic_decay)
    petri_utils.add_arc_from_to(anaerobic, anaerobic_decay, net, weight=10)
    petri_utils.add_arc_from_to(anaerobic_decay, anaerobic, net, weight=100)
    petri_utils.add_arc_from_to(flood, flood_event, net, weight=1)
    petri_utils.add_arc_from_to(aerobic, flood_event, net, weight=500)
    petri_utils.add_arc_from_to(flood_event, anaerobic, net, weight=100)
    
    petri_utils.add_arc_from_to(methanothrix, O2_toxicity, net, weight=5)
    petri_utils.add_arc_from_to(methanosarcina, O2_toxicity, net, weight=2)
    petri_utils.add_arc_from_to(methanocella, O2_toxicity, net, weight=3)
    petri_utils.add_arc_from_to(methanobacterium, O2_toxicity, net, weight=2)
    petri_utils.add_arc_from_to(methanoregula, O2_toxicity, net, weight=2)
    petri_utils.add_arc_from_to(desulfomonile, O2_toxicity, net, weight=2)
    petri_utils.add_arc_from_to(syntrophus, O2_toxicity, net, weight=2)
    petri_utils.add_arc_from_to(aerobic, O2_toxicity, net, weight=100)
    petri_utils.add_arc_from_to(aerobic, O2_consumption, net, weight=100)
    petri_utils.add_arc_from_to(O2_consumption, anaerobic, net, weight=10)
    petri_utils.add_arc_from_to(drip, drain_event, net, weight=1)
    petri_utils.add_arc_from_to(drain_event, drip, net, weight=1)
    petri_utils.add_arc_from_to(drain_event, aerobic, net, weight=200)

    # 8. Phenology & Air Emissions
    petri_utils.add_arc_from_to(CH4_in_soil, tillering, net, weight=1)
    petri_utils.add_arc_from_to(CH4_in_soil, heading, net, weight=1)
    petri_utils.add_arc_from_to(CH4_in_soil, ripening, net, weight=1)
    petri_utils.add_arc_from_to(days21, tillering, net, weight=1)
    petri_utils.add_arc_from_to(days60, heading, net, weight=1)
    petri_utils.add_arc_from_to(days90, ripening, net, weight=1)
    petri_utils.add_arc_from_to(tillering, CH4_in_air, net, weight=tillerno_early)
    petri_utils.add_arc_from_to(heading, CH4_in_air, net, weight=tillerno_mid)
    petri_utils.add_arc_from_to(ripening, CH4_in_air, net, weight=tillerno_late)

    # 9. Methylocaldum Methanotrophy
    petri_utils.add_arc_from_to(CH4_in_soil, methanotrophy, net, weight=1)
    petri_utils.add_arc_from_to(methylocaldum, methanotrophy, net, weight=2)
    petri_utils.add_arc_from_to(aerobic, methanotrophy, net, weight=2)
    petri_utils.add_arc_from_to(methanotrophy, methylocaldum, net, weight=4)
    petri_utils.add_arc_from_to(methanotrophy, co2, net, weight=1)
    petri_utils.add_arc_from_to(CH4_in_soil, aerobic_oxidation, net, weight=1000)
    petri_utils.add_arc_from_to(aerobic, aerobic_oxidation, net, weight=100)
    petri_utils.add_arc_from_to(aerobic_oxidation, aerobic, net, weight=100)
    petri_utils.add_arc_from_to(aerobic_oxidation, co2, net, weight=10)

    return (
        net, CH4_in_soil, CH4_in_air, methanothrix, methanobacterium, methanosarcina,
        methanocella, methanoregula, desulfomonile, syntrophus, sulfuricaulis, nitrospira, methylocaldum,
        organic_matter, acetate, nitrate, n2o, aerobic, n_fertilizer,
        anaerobic, drip, flood, acetogenesis, mg_thrix, mg_bact, mg_sarc, mg_cella, mg_regula,
        syntroph_ferm, desulf_red, sulfuri_ox, anaerobic_decay, sulphate, h2s,
        nitrification, denitrification, tillering, heading, ripening,
        days21, days60, days90, methanotrophy, co2
    )

def run_petrinet(
    ch4mod_df, 
    initial_methanothrix=100, initial_methanobacterium=150, initial_methanosarcina=500,
    initial_methanocella=150, initial_methanoregula=100,
    initial_desulfomonile=100, initial_syntrophus=50, initial_sulfuricaulis=900,
    initial_nitrospira=5000, initial_methylocaldum=750,
    initial_organic_matter=10000, initial_acetate=100, initial_sulphate=10000, initial_n_fertilizer=100000,
    tillerno_early=3, tillerno_mid=10, tillerno_late=20,
    inj1_enable=False, inj1_day=60, inj1_mom=0, inj1_mn=0, inj1_mrt=0, 
    inj2_enable=False, inj2_day=350, inj2_mom=0, inj2_mn=0, inj2_mrt=0, 
    inj3_enable=False, inj3_day=520, inj3_mom=0, inj3_mn=0, inj3_mrt=0,
    harvest_days=None, seasons=None,
    *args, **kwargs
):
    if harvest_days is None: harvest_days = []
    if seasons is None: seasons = []

    (
        net, CH4_in_soil, CH4_in_air, methanothrix, methanobacterium, methanosarcina,
        methanocella, methanoregula, desulfomonile, syntrophus, sulfuricaulis, nitrospira, methylocaldum,
        organic_matter, acetate, nitrate, n2o, aerobic, n_fertilizer,
        anaerobic, drip, flood, acetogenesis, mg_thrix, mg_bact, mg_sarc, mg_cella, mg_regula,
        syntroph_ferm, desulf_red, sulfuri_ox, anaerobic_decay, sulphate, h2s,
        nitrification, denitrification, tillering, heading, ripening,
        days21, days60, days90, methanotrophy, co2
    ) = create_ecosystem_net(tillerno_early, tillerno_mid, tillerno_late)

    marking = pm4py.objects.petri_net.obj.Marking()
    if initial_methanothrix > 0: marking[methanothrix] = initial_methanothrix
    if initial_methanobacterium > 0: marking[methanobacterium] = initial_methanobacterium
    if initial_methanosarcina > 0: marking[methanosarcina] = initial_methanosarcina
    if initial_methanocella > 0: marking[methanocella] = initial_methanocella
    if initial_methanoregula > 0: marking[methanoregula] = initial_methanoregula
    if initial_desulfomonile > 0: marking[desulfomonile] = initial_desulfomonile
    if initial_syntrophus > 0: marking[syntrophus] = initial_syntrophus
    if initial_sulfuricaulis > 0: marking[sulfuricaulis] = initial_sulfuricaulis
    if initial_nitrospira > 0: marking[nitrospira] = initial_nitrospira
    if initial_methylocaldum > 0: marking[methylocaldum] = initial_methylocaldum
    if initial_organic_matter > 0: marking[organic_matter] = initial_organic_matter
    if initial_acetate > 0: marking[acetate] = initial_acetate
    if initial_sulphate > 0: marking[sulphate] = initial_sulphate
    if initial_n_fertilizer > 0: marking[n_fertilizer] = initial_n_fertilizer
        
    marking[days21] = 0
    marking[days60] = 0
    marking[days90] = 0

    current_marking = marking
    history = []
    simulation_steps = len(ch4mod_df) * 24

    def corr_to_weight(r_val):
        return max(0.10, 0.40 + float(r_val))

    for step in range(simulation_steps):
        current_day = min(step // 24, len(ch4mod_df) - 1)
        current_day_num = current_day + 1
        
        # Mid-season additions
        if inj1_enable and step == (inj1_day - 1) * 24:
            if inj1_mom > 0: current_marking[organic_matter] = current_marking.get(organic_matter, 0) + inj1_mom
            if inj1_mn > 0: current_marking[n_fertilizer] = current_marking.get(n_fertilizer, 0) + inj1_mn
            if inj1_mrt > 0: current_marking[methylocaldum] = current_marking.get(methylocaldum, 0) + inj1_mrt
            
        if inj2_enable and step == (inj2_day - 1) * 24:
            if inj2_mom > 0: current_marking[organic_matter] = current_marking.get(organic_matter, 0) + inj2_mom
            if inj2_mn > 0: current_marking[n_fertilizer] = current_marking.get(n_fertilizer, 0) + inj2_mn
            if inj2_mrt > 0: current_marking[methylocaldum] = current_marking.get(methylocaldum, 0) + inj2_mrt
            
        if inj3_enable and step == (inj3_day - 1) * 24:
            if inj3_mom > 0: current_marking[organic_matter] = current_marking.get(organic_matter, 0) + inj3_mom
            if inj3_mn > 0: current_marking[n_fertilizer] = current_marking.get(n_fertilizer, 0) + inj3_mn
            if inj3_mrt > 0: current_marking[methylocaldum] = current_marking.get(methylocaldum, 0) + inj3_mrt

        if current_day_num in harvest_days and step % 24 == 23:
            if CH4_in_soil in current_marking: del current_marking[CH4_in_soil]
            if n2o in current_marking: del current_marking[n2o]
                
        is_tillering = False
        is_heading = False
        is_ripening = False
        in_season = False
        current_doy_in_season = 0

        for (tgd, tgmax, tgend) in seasons:
            if tgd <= current_day_num <= tgend:
                in_season = True
                current_doy_in_season = current_day_num - tgd + 1
                midpoint = tgd + (tgmax - tgd) / 2.0
                if current_day_num < midpoint: is_tillering = True
                elif current_day_num <= tgmax: is_heading = True
                else: is_ripening = True
                break
        
        if is_tillering: current_marking[days21] = 1
        elif days21 in current_marking: del current_marking[days21]

        if is_heading: current_marking[days60] = 1
        elif days60 in current_marking: del current_marking[days60]

        if is_ripening: current_marking[days90] = 1
        elif days90 in current_marking: del current_marking[days90]
        
        # Determine active week stage (Weeks 3, 6, 10)
        if in_season:
            if current_doy_in_season <= 28: active_week = 3
            elif current_doy_in_season <= 56: active_week = 6
            elif current_doy_in_season <= 77: active_week = 10
        else:
            active_week = 0

        r_ch4 = WEEKLY_CORR_DATA[active_week]["CH4"]

        # Physical Redox State
        eh_val = ch4mod_df.iloc[current_day]["Eh_Redox"]
        is_drip_mode = eh_val > -100.0

        if is_drip_mode:
            current_marking[drip] = 1
            if flood in current_marking: del current_marking[flood]
            current_marking[aerobic] = 1000
            if anaerobic in current_marking: del current_marking[anaerobic]
        else:
            if drip in current_marking: del current_marking[drip]
            current_marking[flood] = 1
            # Keep rhizosphere/surface oxic microzone alive in Flood
            current_marking[aerobic] = 150 if in_season else 60
            current_marking[anaerobic] = 100

        anaerobic_ratio = 1.0 / (1.0 + np.exp((eh_val - (-100.0)) / 30.0))
        aerobic_ratio = 1.0 - anaerobic_ratio
        effective_oxic = max(aerobic_ratio, 0.35 if (not is_drip_mode and in_season) else 0.15)
        
        sulphate_count = current_marking.get(sulphate, 0)
        sulfate_availability = sulphate_count / (sulphate_count + 1000.0)
        methanogen_sulfate_suppression = 1.0 - (0.4 * sulfate_availability)

        # Smooth positive dynamic weights
        thrix_r   = corr_to_weight(r_ch4["Methanothrix"]["r"])
        bact_r    = corr_to_weight(r_ch4["Methanobacterium"]["r"])
        sarc_r    = corr_to_weight(r_ch4["Methanosarcina"]["r"])
        cella_r   = corr_to_weight(r_ch4["Methanocella"]["r"])
        regula_r  = corr_to_weight(r_ch4["Methanoregula"]["r"])
        syntro_r  = corr_to_weight(r_ch4["Syntrophus"]["r"])
        desulf_r  = corr_to_weight(r_ch4["Desulfomonile"]["r"])
        methyl_r  = corr_to_weight(r_ch4["Methylocaldum"]["r"])
        sulfuri_r = corr_to_weight(r_ch4["Sulfuricaulis"]["r"])
        nitro_r   = corr_to_weight(-r_ch4["Nitrospira"]["r"])

        sum_mg = thrix_r + bact_r + sarc_r + cella_r + regula_r
        w_thrix  = thrix_r / sum_mg
        w_bact   = bact_r / sum_mg
        w_sarc   = sarc_r / sum_mg
        w_cella  = cella_r / sum_mg
        w_regula = regula_r / sum_mg

        for arc in net.arcs:
            if arc.source == mg_sarc and arc.target == methanosarcina:
                arc.weight = max(4, int(round(3.5 + 8.0 * anaerobic_ratio * methanogen_sulfate_suppression * w_sarc)))
            elif arc.source == mg_bact and arc.target == methanobacterium:
                arc.weight = max(2, int(round(2.0 + 3.5 * anaerobic_ratio * w_bact)))
            elif arc.source == mg_cella and arc.target == methanocella:
                arc.weight = max(2, int(round(2.0 + 3.5 * anaerobic_ratio * w_cella)))
            elif arc.source == mg_thrix and arc.target == methanothrix:
                arc.weight = max(4, int(round(3.8 + 2.5 * anaerobic_ratio * methanogen_sulfate_suppression * w_thrix)))
            elif arc.source == mg_regula and arc.target == methanoregula:
                arc.weight = max(2, int(round(1.8 + 2.5 * anaerobic_ratio * w_regula)))
            elif arc.source == syntroph_ferm and arc.target == syntrophus:
                arc.weight = max(2, int(round(2.0 + 1.5 * anaerobic_ratio * syntro_r)))
            elif arc.source == desulf_red and arc.target == desulfomonile:
                arc.weight = max(2, int(round(2.0 + 2.5 * anaerobic_ratio * sulfate_availability * desulf_r)))
            elif arc.target == sulfuricaulis:
                arc.weight = max(3, int(round(2.5 + 3.0 * (0.4 + 0.6 * effective_oxic) * sulfuri_r * Y_SULFURICAULIS)))
            elif arc.source == methanotrophy and arc.target == methylocaldum:
                arc.weight = max(3, int(round(2.5 + 4.5 * effective_oxic * methyl_r * Y_METHYLOC)))
            elif arc.source == acetogenesis and arc.target == acetate:
                arc.weight = max(10, int(round(40 * (0.2 + anaerobic_ratio * Y_ACETATE))))
            elif arc.source == nitrification and arc.target == nitrospira:
                arc.weight = max(4, int(round(3.5 + 7.5 * effective_oxic * (0.6 + 0.4 * nitro_r) * Y_NITROSPIRA)))

        if step % 24 == 0:
            prod_ch4 = ch4mod_df.iloc[current_day]["Prod_CH4"]
            added_ch4 = int(max(0, prod_ch4) * 10) 
            if added_ch4 > 0:
                current_marking[CH4_in_soil] = current_marking.get(CH4_in_soil, 0) + added_ch4

        state = { 
            "Step": step, 
            "Day": current_day_num,
            "Active_Week": active_week,
            "CH4_in_air": current_marking.get(CH4_in_air, 0),
            "Methanothrix": current_marking.get(methanothrix, 0),
            "Methanobacterium": current_marking.get(methanobacterium, 0),
            "Methanosarcina": current_marking.get(methanosarcina, 0),
            "Methanocella": current_marking.get(methanocella, 0),
            "Methanoregula": current_marking.get(methanoregula, 0),
            "Desulfomonile": current_marking.get(desulfomonile, 0),
            "Syntrophus": current_marking.get(syntrophus, 0),
            "Sulfuricaulis": current_marking.get(sulfuricaulis, 0),
            "Nitrospira": current_marking.get(nitrospira, 0),
            "Methylocaldum": current_marking.get(methylocaldum, 0), 
            "CO2": current_marking.get(co2, 0),
            "N2O": current_marking.get(n2o, 0),
            "Nitrate": current_marking.get(nitrate, 0), 
            "Aerobic": current_marking.get(aerobic, 0), 
            "Organic_Matter": current_marking.get(organic_matter, 0), 
            "Sulphate": current_marking.get(sulphate, 0), 
            "Acetate": current_marking.get(acetate, 0), 
            "N Fertilizer": current_marking.get(n_fertilizer, 0),
            "Stoich_Anaerobic_Ratio": round(anaerobic_ratio, 4)
        }
        history.append(state)

        enabled = list(enabled_transitions(net, current_marking))
        if is_drip_mode:
            for t_anaerobic in [
                acetogenesis, syntroph_ferm, desulf_red, anaerobic_decay, 
                mg_thrix, mg_bact, mg_sarc, mg_cella, mg_regula,
                tillering, heading, ripening
            ]:
                if t_anaerobic in enabled: 
                    enabled.remove(t_anaerobic)
        else:
            enabled = [t for t in enabled if t.name != "O2_Toxicity"]

        if enabled:
            transition_to_fire = random.choice(enabled)
            current_marking = execute(transition_to_fire, net, current_marking)

    results_df = pd.DataFrame(history)
    csv_path = os.path.join("temp_images", "simulation_data.csv")
    results_df.to_csv(csv_path, index=False)

    plot_definitions = [
        ("methanothrix", "Methanothrix", "Methanothrix", "#008080"),
        ("methanobacterium", "Methanobacterium", "Methanobacterium", "#2E8B57"),
        ("methanosarcina", "Methanosarcina", "Methanosarcina", "#228B22"),
        ("methanocella", "Methanocella (Rice Cluster I)", "Methanocella", "#3CB371"),
        ("methanoregula", "Methanoregula", "Methanoregula", "#66CDAA"),
        ("desulfomonile", "Desulfomonile", "Desulfomonile", "#4B0082"),
        ("syntrophus", "Syntrophus", "Syntrophus", "#8A2BE2"),
        ("sulfuricaulis", "Sulfuricaulis", "Sulfuricaulis", "#DAA520"),
        ("nitrospira", "Nitrospira (Nitrifier)", "Nitrospira", "blue"),
        ("methylocaldum", "Methylocaldum (Methanotroph)", "Methylocaldum", "#20B2AA"),
        ("acetate", "Acetate Pool", "Acetate", "brown"),
        ("sulphate", "Sulphate Pool", "Sulphate", "purple"),
        ("nitrate", "Nitrate Pool", "Nitrate", "orange"),
        ("organic_matter", "Organic Matter Pool", "Organic_Matter", "#8B4513"),
        ("n_fertilizer", "N Fertilizer State", "N Fertilizer", "grey"),
        ("aerobic", "Aerobic State (O2)", "Aerobic", "red"),
        ("co2", "CO2 Production", "CO2", "black"),
        ("n2o", "N2O Gas Production", "N2O", "#FF1493"),
        ("ch4_in_air", "Final Emitted CH4 in Air", "CH4_in_air", "red")
    ]
    
    plot_paths = {}
    for key, title, col_name, color in plot_definitions:
        fig, ax = plt.subplots()
        ax.plot(results_df["Step"] / 24, results_df[col_name], label=title, color=color, linewidth=1.5)
        ax.axvline(x=365, color="gray", linestyle=":", alpha=0.7, label="Year 1/2 Cutoff")
        ax.set_title(f"{title} Dynamics (2 Years)")
        ax.set_xlabel("Days (1 - 730)")
        ax.set_ylabel("Token Count")
        ax.grid(True)
        ax.legend()
        path = os.path.join("temp_images", f"{key}_plot.png")
        fig.savefig(path)
        plt.close(fig)
        plot_paths[key] = path

    gviz = pn_visualizer.apply(net, current_marking, parameters={"debug": True})
    net_path = os.path.join("temp_images", "final_net_state.png")
    pn_visualizer.save(gviz, net_path)
    
    return (
        plot_paths["methanothrix"], plot_paths["methanobacterium"], plot_paths["methanosarcina"],
        plot_paths["methanocella"], plot_paths["methanoregula"],
        plot_paths["desulfomonile"], plot_paths["syntrophus"], plot_paths["sulfuricaulis"],
        plot_paths["nitrospira"], plot_paths["methylocaldum"],
        plot_paths["acetate"], plot_paths["sulphate"], plot_paths["nitrate"],
        plot_paths["organic_matter"], plot_paths["n_fertilizer"], plot_paths["aerobic"],
        plot_paths["co2"], plot_paths["n2o"], plot_paths["ch4_in_air"],
        net_path, results_df, csv_path
    )

@spaces.GPU
def run_coupled_simulation(
    tsoil_file, wd_file, tair_file, wmax, dsom, vi, r_growth,
    omn_input, oms_input, sand_fraction, 
    s1_enable, s1_tgd, s1_tgmax, s1_tgend, s2_enable, s2_tgd, s2_tgmax, s2_tgend,
    s3_enable, s3_tgd, s3_tgmax, s3_tgend, wd1_start, wd1_end, wd1_val,
    wd2_start, wd2_end, wd2_val, wd3_start, wd3_end, wd3_val,
    tair_offset, wd_offset,
    initial_methanothrix, initial_methanobacterium, initial_methanosarcina,
    initial_methanocella, initial_methanoregula,
    initial_desulfomonile, initial_syntrophus, initial_sulfuricaulis,
    initial_nitrospira, initial_methylocaldum,
    initial_organic_matter, initial_acetate, initial_sulphate, initial_n_fertilizer,
    tillerno_early, tillerno_mid, tillerno_late,
    inj1_enable, inj1_day, inj1_omn, inj1_oms, inj1_mom, inj1_mn, inj1_mrt,
    inj2_enable, inj2_day, inj2_omn, inj2_oms, inj2_mom, inj2_mn, inj2_mrt,
    inj3_enable, inj3_day, inj3_omn, inj3_oms, inj3_mom, inj3_mn, inj3_mrt,
    *args, **kwargs
):
    harvest_days = []
    seasons = []
    if s1_enable: 
        harvest_days.append(int(s1_tgend))
        seasons.append((int(s1_tgd), int(s1_tgmax), int(s1_tgend)))
    if s2_enable: 
        harvest_days.append(int(s2_tgend))
        seasons.append((int(s2_tgd), int(s2_tgmax), int(s2_tgend)))
    if s3_enable: 
        harvest_days.append(int(s3_tgend))
        seasons.append((int(s3_tgd), int(s3_tgmax), int(s3_tgend)))
        
    fig_ch4, ch4_df, ch4_out_path = run_ch4mod(
        tsoil_file=tsoil_file, wd_file=wd_file, tair_file=tair_file, 
        wmax=wmax, dsom=dsom, vi=vi, r_growth=r_growth,
        omn_input=omn_input, oms_input=oms_input, sand_fraction=sand_fraction, 
        s1_enable=s1_enable, s1_tgd=s1_tgd, s1_tgmax=s1_tgmax, s1_tgend=s1_tgend, 
        s2_enable=s2_enable, s2_tgd=s2_tgd, s2_tgmax=s2_tgmax, s2_tgend=s2_tgend, 
        s3_enable=s3_enable, s3_tgd=s3_tgd, s3_tgmax=s3_tgmax, s3_tgend=s3_tgend, 
        wd1_start=wd1_start, wd1_end=wd1_end, wd1_val=wd1_val, 
        wd2_start=wd2_start, wd2_end=wd2_end, wd2_val=wd2_val, 
        wd3_start=wd3_start, wd3_end=wd3_end, wd3_val=wd3_val, 
        tair_offset=tair_offset, wd_offset=wd_offset, 
        inj1_enable=inj1_enable, inj1_day=inj1_day, inj1_omn=inj1_omn, inj1_oms=inj1_oms, 
        inj2_enable=inj2_enable, inj2_day=inj2_day, inj2_omn=inj2_omn, inj2_oms=inj2_oms, 
        inj3_enable=inj3_enable, inj3_day=inj3_day, inj3_omn=inj3_omn, inj3_oms=inj3_oms
    )
    
    # Explicit keyword argument call prevents any positional shift/overflow
    (
        m_thrix_plot, m_bact_plot, m_sarc_plot, m_cella_plot, m_regula_plot,
        desulf_plot, syntroph_plot, sulfuri_plot,
        nitrospira_plot, methyloc_plot,
        acetate_plot, sulphate_plot, nitrate_plot,
        organic_matter_plot, n_fertilizer_plot, aerobic_plot,
        co2_plot, n2o_plot, ch4_air_plot,
        net_path, pn_df, pn_csv_path
    ) = run_petrinet(
        ch4mod_df=ch4_df, 
        initial_methanothrix=initial_methanothrix,
        initial_methanobacterium=initial_methanobacterium,
        initial_methanosarcina=initial_methanosarcina,
        initial_methanocella=initial_methanocella,
        initial_methanoregula=initial_methanoregula,
        initial_desulfomonile=initial_desulfomonile,
        initial_syntrophus=initial_syntrophus,
        initial_sulfuricaulis=initial_sulfuricaulis,
        initial_nitrospira=initial_nitrospira,
        initial_methylocaldum=initial_methylocaldum,
        initial_organic_matter=initial_organic_matter,
        initial_acetate=initial_acetate,
        initial_sulphate=initial_sulphate,
        initial_n_fertilizer=initial_n_fertilizer,
        tillerno_early=tillerno_early,
        tillerno_mid=tillerno_mid,
        tillerno_late=tillerno_late,
        inj1_enable=inj1_enable,
        inj1_day=inj1_day,
        inj1_mom=inj1_mom,
        inj1_mn=inj1_mn,
        inj1_mrt=inj1_mrt,
        inj2_enable=inj2_enable,
        inj2_day=inj2_day,
        inj2_mom=inj2_mom,
        inj2_mn=inj2_mn,
        inj2_mrt=inj2_mrt,
        inj3_enable=inj3_enable,
        inj3_day=inj3_day,
        inj3_mom=inj3_mom,
        inj3_mn=inj3_mn,
        inj3_mrt=inj3_mrt,
        harvest_days=harvest_days,
        seasons=seasons
    )

    return (
        fig_ch4, ch4_df, ch4_out_path,
        m_thrix_plot, m_bact_plot, m_sarc_plot, m_cella_plot, m_regula_plot,
        desulf_plot, syntroph_plot, sulfuri_plot,
        nitrospira_plot, methyloc_plot,
        acetate_plot, sulphate_plot, nitrate_plot,
        organic_matter_plot, n_fertilizer_plot, aerobic_plot,
        co2_plot, n2o_plot, ch4_air_plot,
        net_path, pn_df, pn_csv_path
    )

# ==========================================
# 3. GRADIO UI DESIGN 
# ==========================================
with gr.Blocks() as app:
    gr.Markdown("# 🌾 Coupled CH4MOD & Microbiome Digital Twin Simulator")
    gr.Markdown(
        "Simulates greenhouse gas dynamics using **CH4MOD** model and a **Petri Net** "
        "driven by stage-dependent empirical field correlation metrics for 10 distinct microbial taxa."
    )
    
    with gr.Tabs():
        with gr.Tab("Crop Timelines (Days 1–730)"):
            gr.Markdown(
                "**2-Year Timeline:**  \n"
                "- **Year 1:** Days 1 to 365 (e.g., Nov: ~305–334, Dec: ~335–365)  \n"
                "- **Year 2:** Days 366 to 730 (e.g., Overwinter Harvest: ~440)"
            )
            with gr.Row():
                with gr.Column(variant="panel"):
                    s1_enable = gr.Checkbox(value=True, label="✅ Season 1 (Spring/Summer Y1)")
                    s1_tgd = gr.Number(value=30, label="Planting Day")
                    s1_tgmax = gr.Number(value=90, label="Max Growth Day")
                    s1_tgend = gr.Number(value=140, label="Harvest Day")
                with gr.Column(variant="panel"):
                    s2_enable = gr.Checkbox(value=True, label="✅ Season 2 (Winter Crop Y1 -> Y2)")
                    s2_tgd = gr.Number(value=320, label="Planting Day (Nov Y1)")
                    s2_tgmax = gr.Number(value=385, label="Max Growth Day (Jan Y2)")
                    s2_tgend = gr.Number(value=440, label="Harvest Day (Mar Y2)")
                with gr.Column(variant="panel"):
                    s3_enable = gr.Checkbox(value=True, label="✅ Season 3 (Summer Crop Y2)")
                    s3_tgd = gr.Number(value=480, label="Planting Day")
                    s3_tgmax = gr.Number(value=550, label="Max Growth Day")
                    s3_tgend = gr.Number(value=610, label="Harvest Day")

        with gr.Tab("CH4MOD Parameters"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Data Uploads (Continuous 730 Days)")
                    tsoil_input = gr.File(label="Soil Temperature File (.txt)")
                    wd_input = gr.File(label="Water Depth File (.txt)")
                    tair_input = gr.File(label="Air Temperature File (.txt)")
                with gr.Column():
                    gr.Markdown("### Crop Constants")
                    wmax_in = gr.Number(value=1600.0, label="Peak Biomass per Season (Wmax) [g/m²]")
                    vi_in = gr.Number(value=1, label="Variety Index (VI)")
                    r_growth_in = gr.Number(value=0.04, label="Growth Rate (R)")                   
                    dsom = gr.Number(value=15000.0, label="Baseline Soil OM (DSOM)")
                    omn_input = gr.Number(value=50.0, label="Non-structural OM (OMn) [g/m²]")
                    oms_input = gr.Number(value=100.0, label="Structural OM (OMs) [g/m²]")                   
                    sand_fraction_in = gr.Slider(minimum=0.0, maximum=1.0, value=0.30, step=0.01, label="Soil Sand Fraction (0.0 - 1.0)")            
            with gr.Row():
                gr.Markdown("### Climate Scenarios")
                tair_offset = gr.Slider(minimum=-5.0, maximum=5.0, value=0.0, step=0.5, label="Temp Offset (°C)")
                wd_offset = gr.Slider(minimum=-15.0, maximum=15.0, value=0.0, step=1.0, label="Water Offset (cm)")

        with gr.Tab("Water Management"):
            gr.Markdown("*Baseline water schedule across 730 days if no file is uploaded.*")
            with gr.Row():
                with gr.Column(variant="panel"):
                    gr.Markdown("**Period 1 (Early Year 1)**")
                    wd1_start = gr.Number(value=1, label="Start Day")
                    wd1_end = gr.Number(value=200, label="End Day")
                    wd1_val = gr.Number(value=10.0, label="Water Level (cm)")
                with gr.Column(variant="panel"):
                    gr.Markdown("**Period 2 (Winter Crop Y1 -> Y2)**")
                    wd2_start = gr.Number(value=201, label="Start Day")
                    wd2_end = gr.Number(value=450, label="End Day")
                    wd2_val = gr.Number(value=15.0, label="Water Level (cm)")
                with gr.Column(variant="panel"):
                    gr.Markdown("**Period 3 (Mid/Late Year 2)**")
                    wd3_start = gr.Number(value=451, label="Start Day")
                    wd3_end = gr.Number(value=730, label="End Day")
                    wd3_val = gr.Number(value=-25.0, label="Water Level (cm)")
                    
        with gr.Tab("Mid-Season Additions"):
            gr.Markdown("### Top-Dressing Fertilizer / Field Injections across 2 Years")
            with gr.Row():
                with gr.Column(variant="panel"):
                    inj1_enable = gr.Checkbox(value=False, label="✅ Event 1")
                    inj1_day = gr.Number(value=60, label="Application Day")
                    inj1_omn = gr.Number(value=0.0, label="OMn (g/m²)")
                    inj1_oms = gr.Number(value=0.0, label="OMs (g/m²)")
                    inj1_mom = gr.Number(value=0, label="Micro OM Tokens")
                    inj1_mn = gr.Number(value=0, label="N-Fert Tokens")
                    inj1_mrt = gr.Number(value=0, label="Methylocaldum Tokens")
                
                with gr.Column(variant="panel"):
                    inj2_enable = gr.Checkbox(value=True, label="✅ Event 2 (Overwinter App)")
                    inj2_day = gr.Number(value=350, label="Application Day (Dec Y1)")
                    inj2_omn = gr.Number(value=20.0, label="OMn (g/m²)")
                    inj2_oms = gr.Number(value=40.0, label="OMs (g/m²)")
                    inj2_mom = gr.Number(value=500, label="Micro OM Tokens")
                    inj2_mn = gr.Number(value=1000, label="N-Fert Tokens")
                    inj2_mrt = gr.Number(value=0, label="Methylocaldum Tokens")
                
                with gr.Column(variant="panel"):
                    inj3_enable = gr.Checkbox(value=False, label="✅ Event 3")
                    inj3_day = gr.Number(value=520, label="Application Day (Year 2)")
                    inj3_omn = gr.Number(value=0.0, label="OMn (g/m²)")
                    inj3_oms = gr.Number(value=0.0, label="OMs (g/m²)")
                    inj3_mom = gr.Number(value=0, label="Micro OM Tokens")
                    inj3_mn = gr.Number(value=0, label="N-Fert Tokens")
                    inj3_mrt = gr.Number(value=0, label="Methylocaldum Tokens")
                    
        with gr.Tab("Microbial Populations"):
            gr.Markdown("### 10 Syntrophic Microbes (Dynamic Weights across Weeks 3, 6, and 10)")
            with gr.Row():
                initial_methanothrix = gr.Number(label="Methanothrix (Wk3 r=0.617 ***)", value=100)
                initial_methanobacterium = gr.Number(label="Methanobacterium (Wk3 r=0.514 ***)", value=150)
                initial_methanosarcina = gr.Number(label="Methanosarcina", value=500)
            with gr.Row():
                initial_methanocella = gr.Number(label="Methanocella (Rice Cluster I, Wk3 r=0.530 ***)", value=150)
                initial_methanoregula = gr.Number(label="Methanoregula (Wk6 Spike r=0.621 ***)", value=100)
                initial_syntrophus = gr.Number(label="Syntrophus (Wk6 Spike r=0.562 ***)", value=50)
            with gr.Row():
                initial_desulfomonile = gr.Number(label="Desulfomonile (Wk3 r=0.501 ***)", value=100)
                initial_sulfuricaulis = gr.Number(label="Sulfuricaulis", value=900)
                initial_nitrospira = gr.Number(label="Nitrospira (Nitrifier)", value=5000)
                initial_methylocaldum = gr.Number(label="Methylocaldum (Methanotroph)", value=750)
            with gr.Row():                
                initial_organic_matter = gr.Number(label="Initial Organic Matter", value=10000)
                initial_acetate = gr.Number(label="Initial Acetate", value=100)
                initial_sulphate = gr.Number(label="Initial Sulphate", value=10000)    
                initial_n_fertilizer = gr.Number(label="Initial N Fertilizer", value=100000)
            with gr.Row():
                tillerno_early = gr.Number(label="Tiller No (early)", value=3)
                tillerno_mid = gr.Number(label="Tiller No (mid)", value=10)
                tillerno_late = gr.Number(label="Tiller No (late)", value=20)
            
    run_btn = gr.Button("▶ Run 2-Year Longitudinal Coupled Simulation", variant="primary", size="lg")
    gr.Markdown("---")
    
    with gr.Tabs():
        with gr.Tab("CH4MOD Model"):
            with gr.Row():
                with gr.Column():
                    plot_output = gr.Plot(label="CH4MOD Plots (Days 1 to 730)")
            with gr.Row():
                with gr.Column():
                    df_output = gr.Dataframe(label="Daily Output Table", row_count=(10, "dynamic"))
                    file_output = gr.File(label="Download Full Results (CSV)")

        with gr.Tab("Microbiome Model"):
            gr.Markdown("### Taxon Dynamics Parameterized across Weeks 3, 6, and 10")
            with gr.Row():
                m_thrix_plot = gr.Image(label="Methanothrix")
                m_bact_plot = gr.Image(label="Methanobacterium")
                m_sarc_plot = gr.Image(label="Methanosarcina")
            with gr.Row():
                m_cella_plot = gr.Image(label="Methanocella (Rice Cluster I)")
                m_regula_plot = gr.Image(label="Methanoregula (Wk 6 Surge)")
                syntroph_plot = gr.Image(label="Syntrophus (Wk 6 Surge)")
            with gr.Row():
                desulf_plot = gr.Image(label="Desulfomonile (Sulfate-Reducer)")
                sulfuri_plot = gr.Image(label="Sulfuricaulis (Sulfur Oxidizer)")
                nitrospira_plot = gr.Image(label="Nitrospira (Nitrifier)")
                methyloc_plot = gr.Image(label="Methylocaldum (Methanotroph)")
            with gr.Row():
                acetate_plot = gr.Image(label="Acetate Pool")
                sulphate_plot = gr.Image(label="Sulphate Pool")
                nitrate_plot = gr.Image(label="Nitrate Pool")
                organic_matter_plot = gr.Image(label="Organic Matter Pool")
            with gr.Row():
                n_fertilizer_plot = gr.Image(label="N Fertilizer State")            
                aerobic_plot = gr.Image(label="Aerobic State")
                co2_plot = gr.Image(label="CO2 Production")
            with gr.Row():
                n2o_plot = gr.Image(label="N2O Production (Denitrification)")
                ch4_air_plot = gr.Image(label="Final Emitted CH4 in Air")
            with gr.Row():
                net_output = gr.Image(label="Final Petri Net Marking Map")
            with gr.Row():
                data_output = gr.DataFrame(label="Simulation Hourly Event Log (including Active Week)")
                download_csv = gr.File(label="Download Raw Simulation Data (.csv)", interactive=False)

    run_btn.click(
        fn=run_coupled_simulation,
        inputs=[
            tsoil_input, wd_input, tair_input, wmax_in, dsom, vi_in, r_growth_in,
            omn_input, oms_input, sand_fraction_in,
            s1_enable, s1_tgd, s1_tgmax, s1_tgend, s2_enable, s2_tgd, s2_tgmax, s2_tgend,
            s3_enable, s3_tgd, s3_tgmax, s3_tgend, wd1_start, wd1_end, wd1_val,
            wd2_start, wd2_end, wd2_val, wd3_start, wd3_end, wd3_val,
            tair_offset, wd_offset,
            initial_methanothrix, initial_methanobacterium, initial_methanosarcina,
            initial_methanocella, initial_methanoregula,
            initial_desulfomonile, initial_syntrophus, initial_sulfuricaulis,
            initial_nitrospira, initial_methylocaldum,
            initial_organic_matter, initial_acetate, initial_sulphate, initial_n_fertilizer,
            tillerno_early, tillerno_mid, tillerno_late,
            inj1_enable, inj1_day, inj1_omn, inj1_oms, inj1_mom, inj1_mn, inj1_mrt,
            inj2_enable, inj2_day, inj2_omn, inj2_oms, inj2_mom, inj2_mn, inj2_mrt,
            inj3_enable, inj3_day, inj3_omn, inj3_oms, inj3_mom, inj3_mn, inj3_mrt
        ],
        outputs=[
            plot_output, df_output, file_output, 
            m_thrix_plot, m_bact_plot, m_sarc_plot, m_cella_plot, m_regula_plot,
            desulf_plot, syntroph_plot, sulfuri_plot,
            nitrospira_plot, methyloc_plot,
            acetate_plot, sulphate_plot, nitrate_plot,
            organic_matter_plot, n_fertilizer_plot, aerobic_plot,
            co2_plot, n2o_plot, ch4_air_plot,
            net_output, data_output, download_csv
        ]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        theme=gr.themes.Soft()
    )
