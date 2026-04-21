import streamlit as st
st.write("hello")
st.stop()



import io
import re
import zipfile 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 
from scipy import signal
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet



# =========================================================
# FILE TYPE HELPERS
# =========================================================
def is_excel_file(uploaded_file):
    if uploaded_file is None:
        return False
    name = uploaded_file.name.lower()
    return name.endswith(".xlsx") or name.endswith(".xls")


# =========================================================
# LOADERS
# =========================================================
def load_emg_table(uploaded_file):
    uploaded_file.seek(0)

    if is_excel_file(uploaded_file):
        df_raw = pd.read_excel(uploaded_file, header=None)
        rows = []

        for _, row in df_raw.iterrows():
            vals = []
            for cell in row.tolist():
                if pd.isna(cell):
                    continue
                try:
                    vals.append(float(str(cell).replace(",", ".")))
                except ValueError:
                    continue

            if len(vals) >= 3:
                t, ch1, ch2 = vals[:3]
                rows.append([t, ch1, ch2])

        if len(rows) < 10:
            raise ValueError(
                f"Il file EMG Excel non contiene abbastanza dati numerici validi. "
                f"Righe trovate: {len(rows)}"
            )

        df = pd.DataFrame(rows, columns=["time", "emg_1", "emg_2"])

    else:
        raw = uploaded_file.read()
        text = raw.decode("utf-8", errors="ignore")

        lines = text.splitlines()
        rows = []
        num_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")

        for line in lines:
            s = line.strip()
            if not s:
                continue

            nums = num_pattern.findall(s)

            if len(nums) >= 3:
                try:
                    t = float(nums[0].replace(",", "."))
                    ch1 = float(nums[1].replace(",", "."))
                    ch2 = float(nums[2].replace(",", "."))
                    rows.append([t, ch1, ch2])
                except ValueError:
                    continue

        if len(rows) < 10:
            raise ValueError(
                f"Il file EMG non contiene abbastanza dati numerici validi. "
                f"Righe trovate: {len(rows)}"
            )

        df = pd.DataFrame(rows, columns=["time", "emg_1", "emg_2"])

    df = df.dropna()
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time")

    dt = df["time"].diff()
    df = df[(dt.isna()) | (dt > 0)].copy()

    if len(df) < 10:
        raise ValueError(f"Dopo pulizia restano troppo pochi campioni EMG: {len(df)}")

    return df.reset_index(drop=True)

def load_force_table(uploaded_file):
    uploaded_file.seek(0)

    if is_excel_file(uploaded_file):
        df_raw = pd.read_excel(uploaded_file, header=None)
        rows = []

        for _, row in df_raw.iterrows():
            vals = []
            for cell in row.tolist():
                if pd.isna(cell):
                    continue
                try:
                    vals.append(float(str(cell).replace(",", ".")))
                except ValueError:
                    continue

            if len(vals) >= 3:
                t, f1, f2 = vals[:3]
                if t >= 0:
                    rows.append([t, f1, f2])

        if len(rows) < 10:
            raise ValueError(
                f"File forza Excel non contiene abbastanza dati numerici validi. "
                f"Righe trovate: {len(rows)}"
            )

        df = pd.DataFrame(rows, columns=["time_s", "force_1_N", "force_2_N"])

    else:
        raw = uploaded_file.read()
        text = raw.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        rows = []
        num_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")

        for line in lines:
            s = line.strip()
            if not s:
                continue

            nums = num_pattern.findall(s)

            if len(nums) >= 3:
                try:
                    t = float(nums[0].replace(",", "."))
                    f1 = float(nums[1].replace(",", "."))
                    f2 = float(nums[2].replace(",", "."))
                    if t < 0:
                        continue
                    rows.append([t, f1, f2])
                except ValueError:
                    continue

        if len(rows) < 10:
            raise ValueError(
                f"File forza non contiene abbastanza dati numerici validi. "
                f"Righe trovate: {len(rows)}"
            )

        df = pd.DataFrame(rows, columns=["time_s", "force_1_N", "force_2_N"])

    df = df.dropna()
    df = df.drop_duplicates(subset=["time_s"])
    df = df.sort_values("time_s")

    dt = df["time_s"].diff()
    df = df[(dt.isna()) | (dt > 0)].copy()

    dt = df["time_s"].diff().dropna()
    if len(dt) > 0:
        dt_med = dt.median()
        if np.isfinite(dt_med) and dt_med > 0:
            df = df[
                (df["time_s"].diff().isna()) | (df["time_s"].diff() < 10 * dt_med)
            ].copy()

    if len(df) < 10:
        raise ValueError(f"Dopo pulizia restano troppo pochi campioni forza: {len(df)}")

    return df.reset_index(drop=True)

# =========================================================
# HELPERS
# =========================================================
def show_grouped_dataframe(df, title_prefix=""):
    if df is None or df.empty:
        st.warning("Tabella vuota.")
        return

    # colonne non-metriche utili per i trial
    base_cols = [c for c in ["Group", "TrialIndex", "EMG_File", "Force_File"] if c in df.columns]

    def show_section(section_title, metric_list):
        cols = [c for c in metric_list if c in df.columns]
        if cols:
            st.markdown(f"### {title_prefix}{section_title}")
            st.dataframe(df[base_cols + cols], use_container_width=True)

    show_section("EMG — ampiezza / attivazione", EMG_AMPLITUDE_METRICS)
    show_section("EMG — frequenza / coordinazione", EMG_FREQUENCY_METRICS)
    show_section("Forza", FORCE_METRICS)
    show_section("Timing / acquisizione", TIMING_METRICS)


def infer_fs_from_time(t, unit):
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]

    if len(t) < 2:
        raise ValueError("Vettore tempo troppo corto per stimare la frequenza di campionamento.")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        raise ValueError("Il vettore tempo non è strettamente crescente o non è valido.")

    dt_med = np.median(dt)

    if unit == "ms":
        dt_med = dt_med / 1000.0

    if not np.isfinite(dt_med) or dt_med <= 0:
        raise ValueError(f"Passo temporale non valido: {dt_med}")

    fs = 1.0 / dt_med

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Frequenza di campionamento non valida: {fs}")

    return fs


def select_force_above_threshold(force, threshold_ratio=0.5):
    force = np.asarray(force, dtype=float)
    if len(force) == 0 or np.all(~np.isfinite(force)):
        return np.array([], dtype=float), np.array([], dtype=bool), np.nan

    peak = np.nanmax(force)
    if not np.isfinite(peak):
        return np.array([], dtype=float), np.array([], dtype=bool), np.nan

    thr = threshold_ratio * peak
    mask = np.isfinite(force) & (force >= thr)
    return force[mask], mask, thr


def minmax_series(series):
    s = pd.Series(series, dtype=float)
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)

    vmin = valid.min()
    vmax = valid.max()

    if np.isclose(vmin, vmax):
        return pd.Series(0.5, index=s.index, dtype=float)

    out = (s - vmin) / (vmax - vmin)
    return out


def safe_trial_name(uploaded_file, fallback):
    if uploaded_file is None:
        return fallback
    return uploaded_file.name


def pair_files_by_order(files_a, files_b):
    n = min(len(files_a), len(files_b))
    pairs = []
    for i in range(n):
        pairs.append((files_a[i], files_b[i]))
    return pairs


# =========================================================
# SIGNAL PROCESSING
# =========================================================
def butter_bandpass(x, fs, low=20.0, high=450.0, order=4):
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Fs EMG non valida: {fs}")

    nyq = fs / 2.0
    lowc = low / nyq
    highc = high / nyq

    if lowc <= 0:
        lowc = 1e-6
    if highc >= 1:
        highc = 0.999999
    if lowc >= highc:
        raise ValueError(
            f"Frequenze di filtro non valide: low={low} Hz, high={high} Hz, fs={fs} Hz"
        )

    b, a = signal.butter(order, [lowc, highc], btype="band")
    return signal.filtfilt(b, a, x)


def notch_filter(x, fs, f0=50.0, q=30.0):
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Fs non valida per notch: {fs}")

    nyq = fs / 2.0
    w0 = f0 / nyq

    if w0 <= 0 or w0 >= 1:
        return x

    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, x)


def emg_envelope(x_rect, fs, cutoff=10.0, order=4):
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Fs non valida per inviluppo: {fs}")

    nyq = fs / 2.0
    wc = cutoff / nyq

    if wc <= 0:
        wc = 1e-6
    if wc >= 1:
        wc = 0.999999

    b, a = signal.butter(order, wc, btype="low")
    return signal.filtfilt(b, a, x_rect)


def smooth_force(x, fs, cutoff=10.0, order=4):
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Frequenza di campionamento forza non valida: {fs}")

    nyq = fs / 2.0
    wn = cutoff / nyq

    if wn >= 1.0:
        cutoff = 0.99 * nyq
        wn = cutoff / nyq

    if wn <= 0:
        raise ValueError(f"Cutoff non valida: cutoff={cutoff}, fs={fs}")

    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, x)


# =========================================================
# METRICS
# =========================================================
def rms(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x))))


def iemg(x_rect, fs):
    x_rect = np.asarray(x_rect, dtype=float)
    if len(x_rect) == 0 or fs <= 0:
        return np.nan
    return float(np.sum(x_rect) / fs)


def welch_psd(x, fs):
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return np.array([]), np.array([])
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), 2048))
    return f, pxx


def mdf(f, pxx):
    if len(f) == 0 or len(pxx) == 0:
        return np.nan
    c = np.cumsum(pxx)
    if c[-1] == 0:
        return np.nan
    return float(np.interp(c[-1] / 2, c, f))


def mpf(f, pxx):
    if len(f) == 0 or len(pxx) == 0:
        return np.nan
    s = np.sum(pxx)
    if s == 0:
        return np.nan
    return float(np.sum(f * pxx) / s)


def detect_onset(envelope, fs, baseline_s=0.3, thr_k=3.0, min_dur_ms=30):
    envelope = np.asarray(envelope, dtype=float)
    if len(envelope) < 5:
        return None, np.nan

    n0 = int(baseline_s * fs)
    n0 = max(1, min(n0, len(envelope) - 1))

    base = envelope[:n0]
    mu = float(np.mean(base))
    sd = float(np.std(base))
    thr = mu + thr_k * sd

    above = envelope > thr
    min_dur = max(int((min_dur_ms / 1000.0) * fs), 1)

    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_dur:
            onset_idx = i - run + 1
            return onset_idx, thr

    return None, thr


def rfd(force, fs, onset_idx, window_ms=200):
    if force is None or onset_idx is None or not np.isfinite(fs) or fs <= 0:
        return np.nan

    n = int((window_ms / 1000.0) * fs)
    end = min(onset_idx + n, len(force) - 1)

    if end <= onset_idx:
        return np.nan

    return float((force[end] - force[onset_idx]) / ((end - onset_idx) / fs))


def coeff_var(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    m = np.mean(x)
    if np.isclose(m, 0):
        return np.nan
    return float(np.std(x) / m)


def cci(envelope_a, envelope_b, eps=1e-12):
    envelope_a = np.asarray(envelope_a, dtype=float)
    envelope_b = np.asarray(envelope_b, dtype=float)

    if len(envelope_a) == 0 or len(envelope_b) == 0:
        return np.nan

    n = min(len(envelope_a), len(envelope_b))
    a = envelope_a[:n]
    b = envelope_b[:n]

    num = 2.0 * np.minimum(a, b)
    den = a + b + eps
    return float(np.mean(num / den))


def interpret_results(summary_df):
    text = []

    def trend(name):
        if name not in summary_df.index:
            return None
        delta = summary_df.loc[name, "Delta_%"]
        if not np.isfinite(delta):
            return None
        return delta

    rfd = trend("Force_RFD_200ms")
    force = trend("Force_peak")
    rms = trend("RMS")
    cci_v = trend("CCI")

    if rfd is not None and rfd > 20:
        text.append("✔ Aumento significativo della velocità di sviluppo della forza (RFD) → miglior reclutamento neuromuscolare.")

    if force is not None and force > 10:
        text.append("✔ Incremento della forza massima → adattamento positivo alla vibrazione.")

    if rms is not None and rms > 10:
        text.append("✔ Maggiore attivazione EMG → aumento del drive neurale.")

    if cci_v is not None and cci_v < 0:
        text.append("✔ Riduzione della co-contrazione → miglior efficienza motoria.")

    tremor = trend("Force_Tremor_3_7Hz")
    if tremor is not None and tremor > 20:
        text.append("⚠ Aumento del tremore → possibile aumento del drive centrale.")

    if not text:
        text.append("Nessuna variazione significativa rilevata.")

    return "\n".join(text)


def tremor_power(force_segment, fs, band=(3.0, 7.0)):
    force_segment = np.asarray(force_segment, dtype=float)
    force_segment = force_segment[np.isfinite(force_segment)]

    if len(force_segment) < 16 or not np.isfinite(fs) or fs <= 0:
        return np.nan

    f, pxx = signal.welch(force_segment, fs=fs, nperseg=min(len(force_segment), 1024))
    lo, hi = band
    m = (f >= lo) & (f <= hi)

    if not np.any(m):
        return np.nan

    return float(np.trapz(pxx[m], f[m]))


def sample_entropy(x, m=2, r=None):
    return np.nan

    if len(x) < 20:
        return np.nan

    # decimazione per velocizzare
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]

    sd = np.std(x)
    if r is None:
        r = 0.2 * sd
    if r <= 0:
        return np.nan

    def _count_matches(mm):
        N = len(x)
        count = 0
        for i in range(N - mm):
            template = x[i:i+mm]
            for j in range(i + 1, N - mm + 1):
                window = x[j:j+mm]
                if np.max(np.abs(template - window)) <= r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if B == 0 or A == 0:
        return np.nan

    return float(-np.log(A / B))

# =========================================================
# SINGLE TRIAL ANALYSIS
# =========================================================
def compute_force_metrics(force, fs, threshold_ratio=0.5, mvc_window_ms=500):
    force = np.asarray(force, dtype=float)
    force = force[np.isfinite(force)]

    results = {
        "Force_peak": np.nan,
        "Force_MVC_500ms": np.nan,
        "Force_top5_mean": np.nan,
        "Force_threshold_pct": threshold_ratio * 100.0,
        "Force_threshold_value": np.nan,
        "Force_samples_above_threshold": np.nan,
        "Force_mean_above_50pct": np.nan,
        "Force_CV_above_50pct": np.nan,
        "Force_AUC_above_50pct": np.nan,
        "Force_Tremor_3_7Hz": np.nan,
        "Force_SampEn": np.nan,
    }

    if len(force) < 5 or not np.isfinite(fs) or fs <= 0:
        return results

    force_peak = float(np.nanmax(force))
    results["Force_peak"] = force_peak

    # MVC robusta: massima media su finestra mobile di 500 ms
    window_n = max(1, int(round((mvc_window_ms / 1000.0) * fs)))
    if len(force) >= window_n:
        kernel = np.ones(window_n, dtype=float) / window_n
        moving_mean = np.convolve(force, kernel, mode="valid")
        results["Force_MVC_500ms"] = float(np.nanmax(moving_mean))

    # Media del top 5%
    q95 = np.nanpercentile(force, 95)
    top5 = force[force >= q95]
    if len(top5) > 0:
        results["Force_top5_mean"] = float(np.nanmean(top5))

    # Metriche sopra soglia
    thr = threshold_ratio * force_peak
    results["Force_threshold_value"] = float(thr)

    mask = force >= thr
    force_sel = force[mask]
    results["Force_samples_above_threshold"] = int(np.sum(mask))

    if len(force_sel) > 1000:
        idx = np.linspace(0, len(force_sel) - 1, 1000).astype(int)
        force_sel = force_sel[idx]

    if len(force_sel) > 0:
        results["Force_mean_above_50pct"] = float(np.nanmean(force_sel))

        m = np.nanmean(force_sel)
        if not np.isclose(m, 0):
            results["Force_CV_above_50pct"] = float(np.nanstd(force_sel) / m)

        dt = 1.0 / fs
        results["Force_AUC_above_50pct"] = float(np.nansum(force_sel) * dt)

        results["Force_Tremor_3_7Hz"] = float(tremor_power(force_sel, fs, band=(3.0, 7.0)))
        results["Force_SampEn"] = float(sample_entropy(force_sel, m=2))

    return results

def moving_rms(x, window_n):
    x = np.asarray(x, dtype=float)
    if len(x) < window_n or window_n < 1:
        return np.array([], dtype=float)

    x2 = x ** 2
    kernel = np.ones(window_n, dtype=float) / window_n
    mr = np.convolve(x2, kernel, mode="valid")
    return np.sqrt(mr)


def compute_emg_mvc_reference(emg_signal, fs, window_ms=500):
    """
    Calcola un riferimento MVC EMG robusto come massimo RMS
    su finestra mobile di 500 ms.
    """
    emg_signal = np.asarray(emg_signal, dtype=float)
    emg_signal = emg_signal[np.isfinite(emg_signal)]

    if len(emg_signal) < 10 or not np.isfinite(fs) or fs <= 0:
        return np.nan

    window_n = max(1, int(round((window_ms / 1000.0) * fs)))
    rms_track = moving_rms(emg_signal, window_n)

    if len(rms_track) == 0:
        return np.nan

    return float(np.nanmax(rms_track))


def apply_emg_normalization_to_trials(trial_df, mvc_emg_ref):
    """
    Aggiunge RMS_norm, IEMG_norm e Neuromuscular_Efficiency.
    """
    trial_df = trial_df.copy()

    if not np.isfinite(mvc_emg_ref) or mvc_emg_ref <= 0:
        trial_df["MVC_EMG_ref"] = np.nan
        trial_df["RMS_norm"] = np.nan
        trial_df["IEMG_norm"] = np.nan
        trial_df["Neuromuscular_Efficiency"] = np.nan
        return trial_df

    trial_df["MVC_EMG_ref"] = mvc_emg_ref
    trial_df["RMS_norm"] = (trial_df["RMS"] / mvc_emg_ref) * 100.0
    trial_df["IEMG_norm"] = (trial_df["IEMG"] / mvc_emg_ref) * 100.0

    # Efficienza neuromuscolare: forza prodotta per unità di attivazione normalizzata
    trial_df["Neuromuscular_Efficiency"] = (
        trial_df["Force_MVC_500ms"] / trial_df["RMS_norm"]
    )

    # gestisci divisioni problematiche
    trial_df.loc[~np.isfinite(trial_df["Neuromuscular_Efficiency"]), "Neuromuscular_Efficiency"] = np.nan

    return trial_df




def analyze_trial(
    df_emg,
    df_force,
    ch1,
    ch2,
    time_col,
    unit,
    force_mode,
    hp=20.0,
    lp=450.0,
    use_notch=False,
    use_notch_100=False,
    notch_q=30.0,
    env_cut=10.0,
    baseline_s=0.3,
    thr_k=3.0,
    min_dur_ms=30,
    force_threshold_ratio=0.5,
):
    t = df_emg[time_col].to_numpy(dtype=float)
    fs = infer_fs_from_time(t, unit)
    t_s = t / 1000.0 if unit == "ms" else t

    emg_raw = df_emg[ch1].to_numpy(dtype=float)
    emg = emg_raw - np.mean(emg_raw)


    if use_notch:
        emg = notch_filter(emg, fs, f0=50.0, q=notch_q)
    
    if use_notch_100:
        emg = notch_filter(emg, fs, 100.0, notch_q)

    

    

    

    emg = butter_bandpass(emg, fs, low=hp, high=lp)
    emg_rect = np.abs(emg)
    emg_env = emg_envelope(emg_rect, fs, cutoff=env_cut)

    onset_idx, thr = detect_onset(
        emg_env,
        fs,
        baseline_s=baseline_s,
        thr_k=thr_k,
        min_dur_ms=min_dur_ms
    )

    if onset_idx is None:
        onset_idx = int(np.argmax(emg_env)) if len(emg_env) > 0 else 0

    i0 = onset_idx
    i200 = min(len(emg), onset_idx + int(0.2 * fs))
    p0 = min(len(emg), onset_idx + int(0.5 * fs))
    p1 = min(len(emg), onset_idx + int(1.5 * fs))

    valid_initial = i200 > i0
    valid_plateau = p1 > p0

    emg2_env = None
    if ch2 is not None and ch2 != "(nessuna)" and ch2 in df_emg.columns:
        emg2_raw = df_emg[ch2].to_numpy(dtype=float)
        emg2 = emg2_raw - np.mean(emg2_raw)

        if use_notch:
            emg2 = notch_filter(emg2, fs, f0=50.0, q=notch_q)
        
        if use_notch_100:
            emg2 = notch_filter(emg2, fs, 100.0, notch_q)


        emg2 = butter_bandpass(emg2, fs, low=hp, high=lp)
        emg2_rect = np.abs(emg2)
        emg2_env = emg_envelope(emg2_rect, fs, cutoff=env_cut)

        force_interp = None
    force_raw = None
    force_smooth = None
    fsf = np.nan
    force_mask = None
    force_thr = np.nan

    if df_force is not None and len(df_force) > 5:
        tf = df_force["time_s"].to_numpy(dtype=float)

        dt = np.diff(tf)
        dt = dt[np.isfinite(dt) & (dt > 0)]

        if len(dt) > 0:
            dt_med = np.median(dt)
            if np.isfinite(dt_med) and dt_med > 0:
                fsf = 1.0 / dt_med

                if force_mode == "force_1_N":
                    force_raw = df_force["force_1_N"].to_numpy(dtype=float)

                elif force_mode == "force_2_N":
                    force_raw = df_force["force_2_N"].to_numpy(dtype=float)

                elif force_mode == "push_minus_pull":
                    force_raw = (
                        df_force["force_1_N"].to_numpy(dtype=float)
                        - df_force["force_2_N"].to_numpy(dtype=float)
                    )

                elif force_mode == "push_plus_pull":
                    force_raw = (
                        df_force["force_1_N"].to_numpy(dtype=float)
                        + df_force["force_2_N"].to_numpy(dtype=float)
                    )

                elif force_mode == "abs_push":
                    force_raw = np.abs(df_force["force_1_N"].to_numpy(dtype=float))

                elif force_mode == "abs_pull":
                    force_raw = np.abs(df_force["force_2_N"].to_numpy(dtype=float))

                elif force_mode == "max_of_two":
                    f1 = np.abs(df_force["force_1_N"].to_numpy(dtype=float))
                    f2 = np.abs(df_force["force_2_N"].to_numpy(dtype=float))
                    force_raw = np.maximum(f1, f2)

                else:
                    raise ValueError(f"force_mode non riconosciuto: {force_mode}")

                valid = np.isfinite(tf) & np.isfinite(force_raw)
                tf = tf[valid]
                force_raw = force_raw[valid]

                if len(tf) > 5 and len(force_raw) > 5:
                    nbase = max(1, min(len(force_raw), int(round(0.2 * fsf))))
                    force_d = force_raw - np.mean(force_raw[:nbase])
                    force_smooth = smooth_force(force_d, fsf, cutoff=10.0)
                    force_interp = np.interp(t_s, tf, force_smooth)

    results = {}

    # =========================
    # EMG metrics
    # =========================
    if valid_plateau:
        results["RMS"] = float(rms(emg[p0:p1]))
        results["IEMG"] = float(iemg(emg_rect[p0:p1], fs))
    else:
        results["RMS"] = np.nan
        results["IEMG"] = np.nan

    if valid_initial:
        results["RMS_0_200ms"] = float(rms(emg[i0:i200]))
    else:
        results["RMS_0_200ms"] = np.nan

    f_psd, p_psd = welch_psd(emg, fs)
    results["MDF"] = float(mdf(f_psd, p_psd))
    results["MPF"] = float(mpf(f_psd, p_psd))

        # riferimento MVC EMG trial-level: massimo RMS su finestra mobile di 500 ms
    results["EMG_MVC_500ms_ref"] = float(compute_emg_mvc_reference(emg, fs, window_ms=500))

    if emg2_env is not None and valid_plateau:
        results["CCI"] = float(cci(emg_env[p0:p1], emg2_env[p0:p1]))
    else:
        results["CCI"] = np.nan

    # =========================
    # Force metrics
    # =========================
    if force_interp is not None:
        force_metrics = compute_force_metrics(
            force_interp,
            fs=fs,
            threshold_ratio=force_threshold_ratio,
            mvc_window_ms=500
        )
        results.update(force_metrics)

        # RFD sulla traccia completa interpolata
        if valid_initial:
            results["Force_RFD_200ms"] = float(rfd(force_interp, fs, i0, window_ms=200))
        else:
            results["Force_RFD_200ms"] = np.nan

        # mantieni anche questi alias se vuoi retrocompatibilità col resto del report
        results["Force_mean"] = results["Force_mean_above_50pct"]
        results["Force_CV"] = results["Force_CV_above_50pct"]

        # salva info soglia anche per grafici
        force_thr = results["Force_threshold_value"]
        if np.isfinite(force_thr):
            force_mask = np.isfinite(force_interp) & (force_interp >= force_thr)
        else:
            force_mask = None

    else:
        results["Force_peak"] = np.nan
        results["Force_MVC_500ms"] = np.nan
        results["Force_top5_mean"] = np.nan
        results["Force_threshold_pct"] = float(force_threshold_ratio * 100.0)
        results["Force_threshold_value"] = np.nan
        results["Force_samples_above_threshold"] = np.nan
        results["Force_mean_above_50pct"] = np.nan
        results["Force_CV_above_50pct"] = np.nan
        results["Force_AUC_above_50pct"] = np.nan
        results["Force_Tremor_3_7Hz"] = np.nan
        results["Force_SampEn"] = np.nan
        results["Force_RFD_200ms"] = np.nan

        # alias
        results["Force_mean"] = np.nan
        results["Force_CV"] = np.nan

    return {
        "results": results,
        "t_s": t_s,
        "emg_raw": emg_raw,
        "emg_filt": emg,
        "emg_env": emg_env,
        "force_interp": force_interp,
        "force_raw": force_raw,
        "force_smooth": force_smooth,
        "force_mask": force_mask,
        "force_threshold_value": force_thr,
        "f_psd": f_psd,
        "p_psd": p_psd,
        "onset_idx": onset_idx,
        "thr": thr,
        "fs": fs,
        "fs_force": fsf,
        "plateau_idx": (p0, p1),
    }


# =========================================================
# BATCH ANALYSIS
# =========================================================

def interpret_results(summary_df):
    text = []

    def trend(name):
        if name not in summary_df.index:
            return None
        delta = summary_df.loc[name, "Delta_%"]
        if not np.isfinite(delta):
            return None
        return delta

    rfd = trend("Force_RFD_200ms")
    force = trend("Force_peak")
    rms = trend("RMS")
    cci_v = trend("CCI")

    if rfd is not None and rfd > 20:
        text.append("✔ Aumento significativo della velocità di sviluppo della forza (RFD) → miglior reclutamento neuromuscolare.")

    if force is not None and force > 10:
        text.append("✔ Incremento della forza massima → adattamento positivo alla vibrazione.")

    if rms is not None and rms > 10:
        text.append("✔ Maggiore attivazione EMG → aumento del drive neurale.")

    if cci_v is not None and cci_v < 0:
        text.append("✔ Riduzione della co-contrazione → miglior efficienza motoria.")

    tremor = trend("Force_Tremor_3_7Hz")
    if tremor is not None and tremor > 20:
        text.append("⚠ Aumento del tremore → possibile aumento del drive centrale.")

    if not text:
        text.append("Nessuna variazione significativa rilevata.")

    return "\n".join(text)


def analyze_group(emg_files, force_files, label, config):
    trial_rows = []
    trial_details = []

    pairs = pair_files_by_order(emg_files, force_files)
    progress = st.progress(0, text=f"Analisi {label} in corso...")

    for idx, (emg_file, force_file) in enumerate(pairs, start=1):
        df_emg = load_emg_table(emg_file)
        df_force = load_force_table(force_file) if force_file is not None else None

        detail = analyze_trial(
            df_emg=df_emg,
            df_force=df_force,
            ch1=config["ch1"],
            ch2=config["ch2"],
            time_col=config["time_col"],
            unit=config["unit"],
            force_mode=config["force_mode"],
            hp=config["hp"],
            lp=config["lp"],
            use_notch=config["use_notch"],
            use_notch_100=config["use_notch_100"],
            notch_q=config["notch_q"],
            env_cut=config["env_cut"],
            baseline_s=config["baseline_s"],
            thr_k=config["thr_k"],
            min_dur_ms=config["min_dur_ms"],
            force_threshold_ratio=config["force_threshold_ratio"],
        )

        row = {"Group": label, "TrialIndex": idx, "EMG_File": safe_trial_name(emg_file, f"{label}_EMG_{idx}")}
        row["Force_File"] = safe_trial_name(force_file, f"{label}_FORCE_{idx}") if force_file is not None else ""

        for k, v in detail["results"].items():
            row[k] = v

        row["Fs_EMG"] = detail["fs"]
        row["Fs_Force"] = detail["fs_force"]
        row["Onset_s"] = detail["t_s"][detail["onset_idx"]] if detail["onset_idx"] is not None and detail["onset_idx"] < len(detail["t_s"]) else np.nan
        row["Plateau_start_s"] = detail["t_s"][detail["plateau_idx"][0]] if detail["plateau_idx"][0] < len(detail["t_s"]) else np.nan
        row["Plateau_end_s"] = detail["t_s"][detail["plateau_idx"][1] - 1] if detail["plateau_idx"][1] - 1 < len(detail["t_s"]) else np.nan

        trial_rows.append(row)
        trial_details.append(detail)

        progress.progress(idx / len(pairs), text=f"Analisi {label}: trial {idx}/{len(pairs)}")

    progress.empty()
    trial_df = pd.DataFrame(trial_rows)
    return trial_df, trial_details

def build_summary(pre_df, post_df):
    numeric_pre = pre_df.select_dtypes(include=[np.number]).drop(columns=["TrialIndex"], errors="ignore")
    numeric_post = post_df.select_dtypes(include=[np.number]).drop(columns=["TrialIndex"], errors="ignore")

    pre_mean = numeric_pre.mean(axis=0, numeric_only=True)
    post_mean = numeric_post.mean(axis=0, numeric_only=True)

    summary = pd.concat([pre_mean.rename("PRE_mean"), post_mean.rename("POST_mean")], axis=1)
    summary["Delta"] = summary["POST_mean"] - summary["PRE_mean"]
    summary["Delta_%"] = 100 * summary["Delta"] / summary["PRE_mean"].replace(0, np.nan)

    # Stability index group-level
    needed = ["Force_CV", "Force_Tremor_3_7Hz", "Force_SampEn"]
    if all(k in summary.index for k in needed):
        metrics_df = pd.DataFrame({
            "PRE_mean": [summary.loc["Force_CV", "PRE_mean"], summary.loc["Force_Tremor_3_7Hz", "PRE_mean"], summary.loc["Force_SampEn", "PRE_mean"]],
            "POST_mean": [summary.loc["Force_CV", "POST_mean"], summary.loc["Force_Tremor_3_7Hz", "POST_mean"], summary.loc["Force_SampEn", "POST_mean"]],
        }, index=needed)

        cv_norm = minmax_series(metrics_df.loc["Force_CV"])
        tr_norm = minmax_series(metrics_df.loc["Force_Tremor_3_7Hz"])
        se_norm = minmax_series(metrics_df.loc["Force_SampEn"])

        if np.all(np.isfinite([cv_norm["PRE_mean"], cv_norm["POST_mean"], tr_norm["PRE_mean"], tr_norm["POST_mean"], se_norm["PRE_mean"], se_norm["POST_mean"]])):
            pre_stab = 100 * (0.45 * (1 - cv_norm["PRE_mean"]) + 0.45 * (1 - tr_norm["PRE_mean"]) + 0.10 * se_norm["PRE_mean"])
            post_stab = 100 * (0.45 * (1 - cv_norm["POST_mean"]) + 0.45 * (1 - tr_norm["POST_mean"]) + 0.10 * se_norm["POST_mean"])

            summary.loc["Force_Stability_Index", "PRE_mean"] = pre_stab
            summary.loc["Force_Stability_Index", "POST_mean"] = post_stab
            summary.loc["Force_Stability_Index", "Delta"] = post_stab - pre_stab
            summary.loc["Force_Stability_Index", "Delta_%"] = 100 * ((post_stab - pre_stab) / pre_stab) if pre_stab != 0 else np.nan

    return summary


def create_zip_report(trial_df, summary_df):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trial_results.csv", trial_df.to_csv(index=False))
        zf.writestr("summary_pre_post.csv", summary_df.to_csv())
    bio.seek(0)
    return bio


# =========================================================
# GUIDE
# =========================================================
METRIC_LABELS = {
    "RMS": "EMG RMS",
    "RMS_norm": "EMG RMS normalizzato",
    "IEMG": "EMG integrato (IEMG)",
    "IEMG_norm": "EMG integrato normalizzato",
    "RMS_0_200ms": "EMG RMS 0–200 ms",
    "EMG_MVC_500ms_ref": "Riferimento MVC EMG 500 ms",
    "MVC_EMG_ref": "Riferimento globale MVC EMG",
    "MDF": "EMG Median Frequency",
    "MPF": "EMG Mean Power Frequency",
    "CCI": "Co-contrazione (CCI)",
    "Force_peak": "Forza di picco",
    "Force_MVC_500ms": "MVC forza media 500 ms",
    "Force_top5_mean": "Media top 5% forza",
    "Force_threshold_pct": "Soglia % forza",
    "Force_threshold_value": "Valore soglia forza",
    "Force_samples_above_threshold": "Campioni sopra soglia",
    "Force_mean": "Forza media",
    "Force_mean_above_50pct": "Forza media sopra 50%",
    "Force_CV": "CV forza",
    "Force_CV_above_50pct": "CV forza sopra 50%",
    "Force_AUC_above_50pct": "AUC forza sopra 50%",
    "Force_Tremor_3_7Hz": "Tremore forza 3–7 Hz",
    "Force_SampEn": "Sample Entropy forza",
    "Force_RFD_200ms": "RFD 0–200 ms",
    "Force_Stability_Index": "Indice stabilità forza",
    "Neuromuscular_Efficiency": "Efficienza neuromuscolare",
    "Fs_EMG": "Fs EMG",
    "Fs_Force": "Fs Forza",
    "Onset_s": "Onset (s)",
    "Plateau_start_s": "Inizio plateau (s)",
    "Plateau_end_s": "Fine plateau (s)",
}

EMG_AMPLITUDE_METRICS = [
    "RMS",
    "RMS_norm",
    "IEMG",
    "IEMG_norm",
    "RMS_0_200ms",
    "EMG_MVC_500ms_ref",
    "MVC_EMG_ref",
]

EMG_FREQUENCY_METRICS = [
    "MDF",
    "MPF",
    "CCI",
]

FORCE_METRICS = [
    "Force_peak",
    "Force_MVC_500ms",
    "Force_top5_mean",
    "Force_threshold_pct",
    "Force_threshold_value",
    "Force_samples_above_threshold",
    "Force_mean",
    "Force_mean_above_50pct",
    "Force_CV",
    "Force_CV_above_50pct",
    "Force_AUC_above_50pct",
    "Force_Tremor_3_7Hz",
    "Force_SampEn",
    "Force_RFD_200ms",
    "Force_Stability_Index",
    "Neuromuscular_Efficiency",
]

TIMING_METRICS = [
    "Fs_EMG",
    "Fs_Force",
    "Onset_s",
    "Plateau_start_s",
    "Plateau_end_s",
]



GUIDE_TEXT = """
### Guida rapida

Questa app analizza **più trial PRE** e **più trial POST**, calcola le metriche per ogni trial,
poi calcola **media PRE**, **media POST**, **Delta** e **Delta %**.

#### EMG
- RMS
- RMS_norm (% MVC EMG)
- IEMG
- IEMG_norm (% MVC EMG)
- RMS 0-200 ms
- MDF
- MPF
- CCI
- EMG_MVC_500ms_ref


#### Forza
Le metriche di stabilità vengono calcolate solo nella parte in cui la forza è
**maggiore o uguale a una soglia percentuale del picco** del trial.

Metriche:
- Force_peak
- Force_MVC_500ms
- Force_top5_mean
- Force_mean_above_50pct
- Force_CV_above_50pct
- Force_AUC_above_50pct
- Force_Tremor_3_7Hz
- Force_RFD_200ms
- Force_Stability_Index
- Neuromuscular_Efficiency

#### Accoppiamento file
I file EMG e forza vengono accoppiati **per ordine di caricamento**.
Quindi:
- EMG PRE 1 con Forza PRE 1
- EMG PRE 2 con Forza PRE 2
- ecc.
"""


# =========================================================
# UI
# =========================================================

def fig_to_image(fig, width=500, height=250):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    return img


def create_pdf_report(summary_df, pre_details=None, post_details=None):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    # Titolo
    elements.append(Paragraph("Report EMG + Forza PRE vs POST", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Tabella risultati
    df_print = summary_df.reset_index().copy()
    df_print = df_print.round(3)
    df_print.columns = ["Parametro", "PRE", "POST", "Delta", "Delta %"]
    df_print["Parametro"] = df_print["Parametro"].astype(str).str.replace("_", " ", regex=False)

    data = [df_print.columns.tolist()] + df_print.values.tolist()

    table = Table(data, colWidths=[140, 80, 80, 80, 80])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    guida = """
    <b>GUIDA INTERPRETATIVA</b><br/><br/>

    PRE = media trial prima<br/>
    POST = media trial dopo<br/>
    Delta = differenza assoluta<br/>
    Delta % = variazione percentuale<br/><br/>

    <b>EMG</b><br/>
    RMS = attivazione muscolare assoluta<br/>
    RMS norm = RMS normalizzato sul riferimento MVC EMG PRE (%)<br/>
    IEMG = energia totale del segnale<br/>
    IEMG norm = IEMG normalizzato sul riferimento MVC EMG PRE (%)<br/>
    RMS 0-200 ms = esplosività neurale iniziale<br/>
    MDF / MPF = contenuto spettrale, fatica / efficienza<br/>
    CCI = co-contrazione tra agonista e antagonista<br/>
    MVC EMG ref = riferimento EMG automatico calcolato dai trial PRE come massimo RMS su finestra mobile di 500 ms<br/><br/>

    <b>Forza</b><br/>
    Force peak = forza massima istantanea<br/>
    Force MVC 500 ms = massima media su finestra mobile di 500 ms<br/>
    Force top5 mean = media del top 5% del segnale forza<br/>
    Force mean above 50% = forza media sopra il 50% del picco<br/>
    Force CV above 50% = stabilità della forza sopra soglia<br/>
    Force AUC above 50% = area sotto la curva sopra soglia<br/>
    Force Tremor 3-7 Hz = oscillazioni a bassa frequenza<br/>
    Force RFD 200 ms = rapidità di sviluppo della forza<br/>
    Force Stability Index = indice sintetico di stabilità<br/><br/>

    <b>Efficienza neuromuscolare</b><br/>
    Neuromuscular Efficiency = Force MVC 500 ms / RMS norm<br/>
    Valori più alti indicano maggiore forza prodotta per unità di attivazione EMG normalizzata.<br/><br/>

    <b>Interpretazione generale</b><br/>
    Aumento di forza, RFD, RMS norm ed efficienza suggerisce miglioramento neuromuscolare.<br/>
    Riduzione della CCI suggerisce maggiore efficienza motoria.<br/>
    """
    elements.append(Paragraph(guida, styles["Normal"]))
    elements.append(Spacer(1, 18))

    # Grafici dal primo trial PRE e POST
    if pre_details and post_details and len(pre_details) > 0 and len(post_details) > 0:
        pre0 = pre_details[0]
        post0 = post_details[0]

        # 1. EMG filtrato
        fig = plt.figure(figsize=(7, 3))
        plt.plot(pre0["t_s"], pre0["emg_filt"], label="PRE")
        plt.plot(post0["t_s"], post0["emg_filt"], label="POST")
        plt.title("EMG filtrato PRE vs POST")
        plt.xlabel("Tempo (s)")
        plt.ylabel("uV")
        plt.legend()
        elements.append(Paragraph("Grafico 1. EMG filtrato PRE vs POST", styles["Heading3"]))
        elements.append(fig_to_image(fig, width=500, height=220))
        elements.append(Spacer(1, 12))

        # 2. Inviluppo EMG
        fig = plt.figure(figsize=(7, 3))
        plt.plot(pre0["t_s"], pre0["emg_env"], label="PRE")
        plt.plot(post0["t_s"], post0["emg_env"], label="POST")
        if pre0["onset_idx"] is not None and pre0["onset_idx"] < len(pre0["t_s"]):
            plt.axvline(pre0["t_s"][pre0["onset_idx"]], linestyle="--")
        if post0["onset_idx"] is not None and post0["onset_idx"] < len(post0["t_s"]):
            plt.axvline(post0["t_s"][post0["onset_idx"]], linestyle="--")
        plt.title("Inviluppo EMG PRE vs POST")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Envelope")
        plt.legend()
        elements.append(Paragraph("Grafico 2. Inviluppo EMG PRE vs POST", styles["Heading3"]))
        elements.append(fig_to_image(fig, width=500, height=220))
        elements.append(Spacer(1, 12))

        # 3. PSD
        fig = plt.figure(figsize=(7, 3))
        if len(pre0["f_psd"]) > 0 and len(pre0["p_psd"]) > 0:
            plt.semilogy(pre0["f_psd"], pre0["p_psd"], label="PRE")
        if len(post0["f_psd"]) > 0 and len(post0["p_psd"]) > 0:
            plt.semilogy(post0["f_psd"], post0["p_psd"], label="POST")
        plt.title("PSD PRE vs POST")
        plt.xlabel("Hz")
        plt.ylabel("Power")
        plt.legend()
        elements.append(Paragraph("Grafico 3. PSD PRE vs POST", styles["Heading3"]))
        elements.append(fig_to_image(fig, width=500, height=220))
        elements.append(Spacer(1, 12))

        # 4. Forza
        if pre0["force_interp"] is not None or post0["force_interp"] is not None:
            fig = plt.figure(figsize=(7, 3))
            if pre0["force_interp"] is not None:
                plt.plot(pre0["t_s"], pre0["force_interp"], label="PRE")
                if np.isfinite(pre0["force_threshold_value"]):
                    plt.axhline(pre0["force_threshold_value"], linestyle="--", label="PRE soglia")
            if post0["force_interp"] is not None:
                plt.plot(post0["t_s"], post0["force_interp"], label="POST")
                if np.isfinite(post0["force_threshold_value"]):
                    plt.axhline(post0["force_threshold_value"], linestyle=":", label="POST soglia")
            plt.title("Forza PRE vs POST")
            plt.xlabel("Tempo (s)")
            plt.ylabel("N")
            plt.legend()
            elements.append(Paragraph("Grafico 4. Forza PRE vs POST", styles["Heading3"]))
            elements.append(fig_to_image(fig, width=500, height=220))
            elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.set_page_config(page_title="EMG + Forza Batch PRE vs POST", layout="wide")
st.title("EMG + Forza Batch PRE vs POST")

with st.expander("Guida alle analisi e ai parametri", expanded=False):
    st.markdown(GUIDE_TEXT)

st.header("Caricamento dati batch")

c1, c2 = st.columns(2)

with c1:
    st.subheader("PRE")
    emg_pre_files = st.file_uploader(
        "File EMG PRE",
        type=["csv", "txt", "xlsx", "xls"],
        accept_multiple_files=True,
        key="emg_pre_batch",
    )
    force_pre_files = st.file_uploader(
        "File Forza PRE",
        type=["csv", "txt", "xlsx", "xls"],
        accept_multiple_files=True,
        key="force_pre_batch",
    )

with c2:
    st.subheader("POST")
    emg_post_files = st.file_uploader(
        "File EMG POST",
        type=["csv", "txt", "xlsx", "xls"],
        accept_multiple_files=True,
        key="emg_post_batch",
    )
    force_post_files = st.file_uploader(
        "File Forza POST",
        type=["csv", "txt", "xlsx", "xls"],
        accept_multiple_files=True,
        key="force_post_batch",
    )

if not emg_pre_files or not emg_post_files:
    st.info("Carica almeno i file EMG PRE e POST.")
    st.stop()

st.write(f"Numero file EMG PRE: {len(emg_pre_files)}")
st.write(f"Numero file Forza PRE: {len(force_pre_files) if force_pre_files else 0}")
st.write(f"Numero file EMG POST: {len(emg_post_files)}")
st.write(f"Numero file Forza POST: {len(force_post_files) if force_post_files else 0}")

# Preview using first PRE EMG file
try:
    preview_emg = load_emg_table(emg_pre_files[0])
except Exception as e:
    st.error(f"Errore nel primo file EMG PRE: {e}")
    st.stop()

st.subheader("Preview primo file EMG PRE")
st.dataframe(preview_emg.head(10), use_container_width=True)

time_col = st.selectbox("Colonna tempo EMG", preview_emg.columns.tolist(), index=0)
emg_cols = [c for c in preview_emg.columns if c != time_col]

cc1, cc2 = st.columns(2)
with cc1:
    ch1 = st.selectbox("Canale EMG 1", emg_cols, index=0)
with cc2:
    ch2 = st.selectbox("Canale EMG 2 (opzionale)", ["(nessuna)"] + emg_cols, index=0 if len(emg_cols) < 2 else 2)

unit = st.selectbox("Unità tempo EMG", ["ms", "s"], index=0)

# Sidebar parameters
st.sidebar.header("Pre-processing EMG")
hp = st.sidebar.number_input("High-pass EMG (Hz)", min_value=1.0, max_value=500.0, value=20.0, step=1.0)
lp = st.sidebar.number_input("Low-pass EMG (Hz)", min_value=20.0, max_value=2000.0, value=450.0, step=10.0)
use_notch = st.sidebar.checkbox("Notch 50 Hz", value=False)
use_notch_100 = st.sidebar.checkbox(
    "Notch 100 Hz (vibrazione)", 
    value=False,
    key="notch_100_checkbox"
)
notch_q = st.sidebar.number_input("Notch Q", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
env_cut = st.sidebar.number_input("Cutoff inviluppo (Hz)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)

st.sidebar.header("Onset")
baseline_s = st.sidebar.number_input("Baseline onset (s)", min_value=0.1, max_value=5.0, value=0.3, step=0.1)
thr_k = st.sidebar.number_input("Soglia onset = media + k*std", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
min_dur_ms = st.sidebar.number_input("Durata minima sopra soglia (ms)", min_value=5, max_value=500, value=30, step=5)

st.sidebar.header("Forza")
force_mode = st.sidebar.selectbox(
    "Segnale forza da usare",
    [
        "push_plus_pull",
        "force_1_N",
        "force_2_N",
        "push_minus_pull",
        "abs_push",
        "abs_pull",
        "max_of_two"
    ],
    index=0
)

force_threshold_ratio = st.sidebar.slider(
    "Soglia forza (% del picco)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
) / 100.0

run = st.button("Esegui analisi batch")
if not run:
    st.stop()

config = {
    "time_col": time_col,
    "ch1": ch1,
    "ch2": ch2,
    "unit": unit,
    "force_mode": force_mode,
    "hp": hp,
    "lp": lp,
    "use_notch": use_notch,
    "use_notch_100": use_notch_100,
    "notch_q": notch_q,
    "env_cut": env_cut,
    "baseline_s": baseline_s,
    "thr_k": thr_k,
    "min_dur_ms": min_dur_ms,
    "force_threshold_ratio": force_threshold_ratio,
}

try:
    pre_trial_df, pre_details = analyze_group(
        emg_pre_files,
        force_pre_files if force_pre_files else [None] * len(emg_pre_files),
        "PRE",
        config,
    )
    post_trial_df, post_details = analyze_group(
        emg_post_files,
        force_post_files if force_post_files else [None] * len(emg_post_files),
        "POST",
        config,
    )
except Exception as e:
    st.error(f"Errore durante l'analisi batch: {e}")
    st.stop()

# ==========================================
# Normalizzazione EMG su riferimento MVC PRE
# ==========================================
# uso il massimo riferimento EMG trovato nei trial PRE
mvc_emg_ref_global = np.nanmax(pre_trial_df["EMG_MVC_500ms_ref"].to_numpy(dtype=float))

pre_trial_df = apply_emg_normalization_to_trials(pre_trial_df, mvc_emg_ref_global)
post_trial_df = apply_emg_normalization_to_trials(post_trial_df, mvc_emg_ref_global)

trial_df = pd.concat([pre_trial_df, post_trial_df], axis=0, ignore_index=True)
summary_df = build_summary(pre_trial_df, post_trial_df)

# aggiungo anche il riferimento globale nel summary
summary_df.loc["MVC_EMG_ref", "PRE_mean"] = mvc_emg_ref_global
summary_df.loc["MVC_EMG_ref", "POST_mean"] = mvc_emg_ref_global
summary_df.loc["MVC_EMG_ref", "Delta"] = 0.0
summary_df.loc["MVC_EMG_ref", "Delta_%"] = 0.0

st.subheader("Esporta report")

pdf_buffer = create_pdf_report(summary_df, pre_details, post_details)

st.download_button(
    "📄 Scarica report PDF",
    data=pdf_buffer,
    file_name="report_emg_force_pre_post.pdf",
    mime="application/pdf",
    key="download_pdf_report",
)

st.subheader("Risultati per trial")
show_grouped_dataframe(trial_df)

st.subheader("Summary PRE vs POST")

summary_print = summary_df.reset_index().rename(columns={"index": "Parametro"})

def show_grouped_summary(summary_source_df, section_title, metric_list):
    rows = [m for m in metric_list if m in summary_source_df.index]
    if rows:
        st.markdown(f"### {section_title}")
        st.dataframe(summary_source_df.loc[rows], use_container_width=True)

show_grouped_summary(summary_df, "EMG — ampiezza / attivazione", EMG_AMPLITUDE_METRICS)
show_grouped_summary(summary_df, "EMG — frequenza / coordinazione", EMG_FREQUENCY_METRICS)
show_grouped_summary(summary_df, "Forza", FORCE_METRICS)
show_grouped_summary(summary_df, "Timing / acquisizione", TIMING_METRICS)

st.subheader("📘 Guida interpretativa dei risultati")

st.markdown("""
## 🧠 1. STRUTTURA GENERALE DEL RISULTATO
Hai una tabella con:

- PRE_mean → media dei trial prima  
- POST_mean → media dei trial dopo vibrazione  
- Delta → differenza assoluta  
- Delta_% → variazione percentuale  

👉 Quindi stai già facendo una vera analisi scientifica aggregata.

---

## ⚡ 2. ATTIVAZIONE MUSCOLARE (EMG)

### 🔹 RMS
Misura l’ampiezza del segnale EMG → numero di unità motorie attive.

👉 Un aumento indica maggiore attivazione neurale.

---

### 🔹 IEMG
Energia totale del segnale nel tempo.

👉 Conferma il livello globale di attivazione.

---

### 🔹 RMS_0_200ms
Attivazione nei primi 200 ms → esplosività neurale.

👉 Indice chiave di reclutamento rapido.

---

## 📊 3. DOMINIO FREQUENZA

### 🔹 MDF / MPF
Indicatori di fatica e tipo di attivazione.

👉 Frequenze più alte = minore fatica / maggiore efficienza.

---

## 🤝 4. CO-CONTRAZIONE (CCI)

Misura quanto agonista e antagonista lavorano insieme.

👉 Valori più bassi = maggiore efficienza motoria.

---

## 💪 5. FORZA

⚠️ Analisi effettuata solo sopra il 50% della forza massima.

### 🔹 Force_peak
Forza massima → capacità assoluta

### 🔹 Force_mean
Forza sostenuta → controllo motorio

### 🔹 Force_CV
Variabilità → stabilità della forza

### 🔹 Force_Tremor_3_7Hz
Oscillazioni fisiologiche → output neurale

### 🔹 Force_RFD_200ms
Velocità di sviluppo forza → parametro più importante

---

## 🧠 6. TIMING

### 🔹 Onset
Tempo di attivazione (attenzione: sensibile al threshold)

---

## 🔬 7. INTERPRETAZIONE GLOBALE

✔ aumento forza  
✔ aumento RFD  
✔ aumento attivazione EMG  
✔ riduzione co-contrazione  

👉 effetto tipico della vibrazione:

- facilitazione riflessa (Ia afferents)  
- aumento sincronizzazione unità motorie  
- aumento drive centrale  

---

## 📌 8. PARAMETRI USATI

EMG:
- Band-pass: 20–450 Hz  
- Envelope: 10 Hz  

Onset:
- baseline: 0.3 s  
- soglia: mean + 3·std  

Forza:
- threshold: 50%  
- smoothing: 10 Hz  

---

## 🧭 9. CONCLUSIONE

👉 Risultati coerenti con miglioramento neuromuscolare post vibrazione.
""")

st.subheader("Interpretazione automatica")
st.markdown(interpret_results(summary_df))

use_notch_100 = st.sidebar.checkbox("Notch 100 Hz (vibrazione)", value=False)


# =========================================================
# PLOTS
# =========================================================
st.subheader("Grafici summary")

plot_metrics = [
    "RMS",
    "RMS_norm",
    "IEMG",
    "IEMG_norm",
    "RMS_0_200ms",
    "MDF",
    "MPF",
    "CCI",
    "EMG_MVC_500ms_ref",
    "Force_peak",
    "Force_MVC_500ms",
    "Force_top5_mean",
    "Force_mean",
    "Force_mean_above_50pct",
    "Force_CV",
    "Force_CV_above_50pct",
    "Force_AUC_above_50pct",
    "Force_Tremor_3_7Hz",
    "Force_RFD_200ms",
    "Force_Stability_Index",
    "Neuromuscular_Efficiency",
]

available_metrics = [m for m in plot_metrics if m in summary_df.index]
metric_to_plot = st.selectbox("Metrica da visualizzare", available_metrics, index=0 if available_metrics else None)

if metric_to_plot:
    fig = plt.figure()
    vals = [
        summary_df.loc[metric_to_plot, "PRE_mean"] if "PRE_mean" in summary_df.columns else np.nan,
        summary_df.loc[metric_to_plot, "POST_mean"] if "POST_mean" in summary_df.columns else np.nan,
    ]
    plt.bar(["PRE", "POST"], vals)
    plt.title(metric_to_plot)
    plt.ylabel("Valore")
    st.pyplot(fig, clear_figure=True)

st.subheader("Tracce esempio: primo trial PRE vs primo trial POST")
if len(pre_details) > 0 and len(post_details) > 0:
    pre0 = pre_details[0]
    post0 = post_details[0]

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        fig = plt.figure()
        plt.plot(pre0["t_s"], pre0["emg_filt"], label="PRE", alpha=0.8)
        plt.plot(post0["t_s"], post0["emg_filt"], label="POST", alpha=0.8)
        plt.title("EMG filtrato")
        plt.xlabel("Tempo (s)")
        plt.ylabel("uV")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    with pcol2:
        fig = plt.figure()
        plt.plot(pre0["t_s"], pre0["emg_env"], label="PRE")
        plt.plot(post0["t_s"], post0["emg_env"], label="POST")
        if pre0["onset_idx"] is not None and pre0["onset_idx"] < len(pre0["t_s"]):
            plt.axvline(pre0["t_s"][pre0["onset_idx"]], linestyle="--")
        if post0["onset_idx"] is not None and post0["onset_idx"] < len(post0["t_s"]):
            plt.axvline(post0["t_s"][post0["onset_idx"]], linestyle="--")
        plt.title("Inviluppo EMG")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Envelope")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    pcol3, pcol4 = st.columns(2)

    with pcol3:
        fig = plt.figure()
        if len(pre0["f_psd"]) > 0:
            plt.semilogy(pre0["f_psd"], pre0["p_psd"], label="PRE")
        if len(post0["f_psd"]) > 0:
            plt.semilogy(post0["f_psd"], post0["p_psd"], label="POST")
        plt.title("PSD")
        plt.xlabel("Hz")
        plt.ylabel("Power")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    with pcol4:
        if pre0["force_interp"] is not None or post0["force_interp"] is not None:
            fig = plt.figure()
            if pre0["force_interp"] is not None:
                plt.plot(pre0["t_s"], pre0["force_interp"], label="PRE")
                if np.isfinite(pre0["force_threshold_value"]):
                    plt.axhline(pre0["force_threshold_value"], linestyle="--", label="PRE soglia")
            if post0["force_interp"] is not None:
                plt.plot(post0["t_s"], post0["force_interp"], label="POST")
                if np.isfinite(post0["force_threshold_value"]):
                    plt.axhline(post0["force_threshold_value"], linestyle=":", label="POST soglia")
            plt.title("Forza")
            plt.xlabel("Tempo (s)")
            plt.ylabel("N")
            plt.legend()
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Forza non disponibile nel primo trial PRE/POST.")




# =========================================================
# EXPORT
# =========================================================
zip_buffer = create_zip_report(trial_df, summary_df)

st.download_button(
    "Scarica report completo ZIP",
    data=zip_buffer,
    file_name="report_emg_force_pre_post.zip",
    mime="application/zip",
)

st.download_button(
    "Scarica trial_results.csv",
    data=trial_df.to_csv(index=False).encode("utf-8"),
    file_name="trial_results.csv",
    mime="text/csv",
)

st.download_button(
    "Scarica summary_pre_post.csv",
    data=summary_df.to_csv().encode("utf-8"),
    file_name="summary_pre_post.csv",
    mime="text/csv",
)