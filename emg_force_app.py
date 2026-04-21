import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal


# =========================================================
# LOADERS
# =========================================================
def _guess_sep(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if ";" in sample:
        return ";"
    return ","


def load_emg_table(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)

    # decimali italiani -> punto
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    sample = "\n".join(text.splitlines()[:20])
    sep = _guess_sep(sample)

    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_force_table(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)

    # decimali italiani -> punto
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    lines = text.splitlines()

    # trova la prima riga che inizia con numero
    data_start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^[-+]?\d+(\.\d+)?([;\t,])", s):
            data_start = i
            break

    if data_start is None:
        raise ValueError("Non riesco a trovare l'inizio dei dati numerici nel file forza.")

    first = lines[data_start]
    if ";" in first:
        sep = ";"
    elif "\t" in first:
        sep = "\t"
    else:
        sep = ","

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        header=None,
        skiprows=data_start,
        engine="python"
    )

    # rimuovi colonne completamente vuote
    df = df.dropna(axis=1, how="all")

    # tieni solo prime 3 colonne utili
    if df.shape[1] < 3:
        raise ValueError("Il file forza deve contenere almeno 3 colonne: tempo, forza1, forza2.")

    df = df.iloc[:, :3].copy()
    df.columns = ["time_s", "force_1_N", "force_2_N"]

    # conversione numerica robusta
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # elimina righe non valide
    df = df.dropna(subset=["time_s", "force_1_N", "force_2_N"]).copy()

    # rimuovi righe duplicate o non crescenti nel tempo
    df = df.sort_values("time_s")
    df = df.loc[np.isfinite(df["time_s"])]
    df = df.drop_duplicates(subset=["time_s"])

    # tieni solo tempi strettamente crescenti
    dt = df["time_s"].diff()
    df = df[(dt.isna()) | (dt > 0)].copy()

    if len(df) < 10:
        raise ValueError("Troppi pochi campioni validi nel file forza dopo la pulizia.")

    return df.reset_index(drop=True)

# =========================================================
# HELPERS
# =========================================================
def find_time_column(df: pd.DataFrame):
    candidates = [c for c in df.columns if "time" in c.lower()]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def find_signal_columns(df: pd.DataFrame, exclude_cols):
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def infer_fs_from_time(t: np.ndarray, time_unit: str) -> float:
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Time vector non valido.")
    dt_s = dt / 1000.0 if time_unit == "ms" else dt
    return 1.0 / dt_s


# =========================================================
# SIGNAL PROCESSING
# =========================================================
def butter_bandpass(x, fs, low=20.0, high=450.0, order=4):
    nyq = fs / 2
    lowc = max(low / nyq, 1e-6)
    highc = min(high / nyq, 0.999999)
    b, a = signal.butter(order, [lowc, highc], btype="band")
    return signal.filtfilt(b, a, x)


def notch_filter(x, fs, f0=50.0, q=30.0):
    nyq = fs / 2
    w0 = f0 / nyq
    if w0 <= 0 or w0 >= 1:
        return x
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, x)


def emg_envelope(x_rect, fs, cutoff=10.0, order=4):
    nyq = fs / 2
    wc = min(cutoff / nyq, 0.999999)
    b, a = signal.butter(order, wc, btype="low")
    return signal.filtfilt(b, a, x_rect)


def smooth_force(x, fs, cutoff=10.0, order=4):
    nyq = fs / 2
    wc = min(cutoff / nyq, 0.999999)
    b, a = signal.butter(order, wc, btype="low")
    return signal.filtfilt(b, a, x)


# =========================================================
# METRICS
# =========================================================
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def iemg(x_rect, fs):
    return float(np.sum(x_rect) / fs)


def welch_psd(x, fs):
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), 2048))
    return f, pxx


def mdf(f, pxx):
    c = np.cumsum(pxx)
    if c[-1] == 0:
        return np.nan
    return float(np.interp(c[-1] / 2, c, f))


def mpf(f, pxx):
    s = np.sum(pxx)
    if s == 0:
        return np.nan
    return float(np.sum(f * pxx) / s)


def detect_onset(envelope, fs, baseline_s=0.5, thr_k=3.0, min_dur_ms=30):
    n0 = int(baseline_s * fs)
    n0 = min(n0, len(envelope) - 1)
    base = envelope[:n0]
    mu, sd = float(np.mean(base)), float(np.std(base))
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
    n = int((window_ms / 1000.0) * fs)
    end = min(onset_idx + n, len(force) - 1)
    if end <= onset_idx:
        return np.nan
    return float((force[end] - force[onset_idx]) / ((end - onset_idx) / fs))


def coeff_var(x):
    m = np.mean(x)
    if np.isclose(m, 0):
        return np.nan
    return float(np.std(x) / m)


def cci(envelope_a, envelope_b, eps=1e-12):
    num = 2.0 * np.minimum(envelope_a, envelope_b)
    den = envelope_a + envelope_b + eps
    return float(np.mean(num / den))


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="EMG + Forza Analyzer", layout="wide")
st.title("Analisi EMG + Forza")

colA, colB = st.columns(2)
with colA:
    emg_file = st.file_uploader("File EMG", type=["csv", "txt"])
with colB:
    force_file = st.file_uploader("File Forza", type=["csv", "txt"])

if emg_file is None:
    st.stop()

# ---------------- EMG ----------------
df_emg = load_emg_table(emg_file)

st.subheader("Preview EMG")
st.dataframe(df_emg.head(20), use_container_width=True)

time_col_emg = find_time_column(df_emg)
time_unit_emg = st.selectbox("Unità tempo EMG", ["ms", "s"], index=0)

if time_col_emg is None:
    st.error("Non trovo la colonna tempo nel file EMG.")
    st.stop()

sig_cols_emg = find_signal_columns(df_emg, exclude_cols=[time_col_emg])
if len(sig_cols_emg) < 1:
    st.error("Non trovo colonne numeriche EMG oltre al tempo.")
    st.stop()

emg_ch1_col = st.selectbox("Colonna EMG canale 1", sig_cols_emg, index=0)
emg_ch2_col = st.selectbox("Colonna EMG canale 2", ["(nessuna)"] + sig_cols_emg, index=0)

t_emg = df_emg[time_col_emg].to_numpy(dtype=float)
fs_emg = infer_fs_from_time(t_emg, time_unit_emg)
st.info(f"Fs EMG stimata: {fs_emg:.2f} Hz")

# converto sempre t_emg in secondi
t_emg_s = t_emg / 1000.0 if time_unit_emg == "ms" else t_emg

emg1 = df_emg[emg_ch1_col].to_numpy(dtype=float)
emg2 = None if emg_ch2_col == "(nessuna)" else df_emg[emg_ch2_col].to_numpy(dtype=float)

# ---------------- FORCE ----------------
df_force = None
force = None
t_force_s = None
fs_force = None

if force_file is not None:
    df_force = load_force_table(force_file)

    st.subheader("Preview Forza")
    st.dataframe(df_force.head(20), use_container_width=True)

    t_force_s = df_force["time_s"].to_numpy(dtype=float)

    dt_force = np.diff(t_force_s)
    dt_force = dt_force[np.isfinite(dt_force) & (dt_force > 0)]

    if len(dt_force) == 0:
        st.error("Impossibile stimare Fs della forza: tempo non valido.")
        st.stop()

    dt_med = np.median(dt_force)

    if not np.isfinite(dt_med) or dt_med <= 0:
        st.error("Fs della forza non valida.")
        st.stop()

    fs_force = 1.0 / dt_med
    st.info(f"Fs Forza stimata: {fs_force:.2f} Hz")

    force_mode = st.selectbox(
        "Quale forza usare?",
        ["force_1_N", "force_2_N", "push_minus_pull", "abs_push", "abs_pull", "max_of_two"]
    )

    if force_mode == "force_1_N":
        force = df_force["force_1_N"].to_numpy(dtype=float)
    elif force_mode == "force_2_N":
        force = df_force["force_2_N"].to_numpy(dtype=float)
    elif force_mode == "push_minus_pull":
        force = (df_force["force_1_N"] - df_force["force_2_N"]).to_numpy(dtype=float)
    elif force_mode == "abs_push":
        force = np.abs(df_force["force_1_N"].to_numpy(dtype=float))
    elif force_mode == "abs_pull":
        force = np.abs(df_force["force_2_N"].to_numpy(dtype=float))
    elif force_mode == "max_of_two":
        f1 = np.abs(df_force["force_1_N"].to_numpy(dtype=float))
        f2 = np.abs(df_force["force_2_N"].to_numpy(dtype=float))
        force = np.maximum(f1, f2)
else:
    st.warning("File forza non caricato.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Pre-processing")
hp = st.sidebar.number_input("High-pass EMG (Hz)", value=20.0, min_value=0.0, max_value=200.0, step=1.0)
lp = st.sidebar.number_input("Low-pass EMG (Hz)", value=450.0, min_value=50.0, max_value=1000.0, step=10.0)
use_notch = st.sidebar.checkbox("Notch 50 Hz", value=False)
notch_q = st.sidebar.number_input("Notch Q", value=30.0, min_value=5.0, max_value=100.0, step=1.0)
env_cut = st.sidebar.number_input("Cutoff inviluppo (Hz)", value=10.0, min_value=1.0, max_value=30.0, step=1.0)

st.sidebar.header("Onset")
baseline_s = st.sidebar.number_input("Baseline onset (s)", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
thr_k = st.sidebar.number_input("Soglia onset: mean + k*std", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
min_dur_ms = st.sidebar.number_input("Durata minima sopra soglia (ms)", value=30, min_value=5, max_value=200, step=5)

st.sidebar.header("Analisi")
do_time = st.sidebar.checkbox("RMS / IEMG", value=True)
do_freq = st.sidebar.checkbox("MDF / MPF", value=True)
do_onset = st.sidebar.checkbox("Onset", value=True)
do_cci = st.sidebar.checkbox("CCI", value=True)
do_force = st.sidebar.checkbox("Metriche forza", value=True)

run = st.button("Esegui analisi")
if not run:
    st.stop()

# =========================================================
# PROCESSING
# =========================================================
x1 = emg1 - np.mean(emg1)
if use_notch:
    x1 = notch_filter(x1, fs_emg, 50.0, notch_q)
x1 = butter_bandpass(x1, fs_emg, hp, lp)
x1_rect = np.abs(x1)
x1_env = emg_envelope(x1_rect, fs_emg, env_cut)

x2_env = None
if emg2 is not None:
    x2 = emg2 - np.mean(emg2)
    if use_notch:
        x2 = notch_filter(x2, fs_emg, 50.0, notch_q)
    x2 = butter_bandpass(x2, fs_emg, hp, lp)
    x2_rect = np.abs(x2)
    x2_env = emg_envelope(x2_rect, fs_emg, env_cut)

onset_idx = None
thr = None
if do_onset:
    onset_idx, thr = detect_onset(x1_env, fs_emg, baseline_s, thr_k, min_dur_ms)

plateau_start = int((onset_idx if onset_idx is not None else 0) + 1.0 * fs_emg)
plateau_end = int((onset_idx if onset_idx is not None else 0) + 2.0 * fs_emg)
plateau_start = np.clip(plateau_start, 0, len(x1_env) - 1)
plateau_end = np.clip(plateau_end, plateau_start + 1, len(x1_env))

force_on_emg = None
force_s = None

if force is not None:
    if fs_force is None or not np.isfinite(fs_force):
        st.error("Fs forza non valida.")
        st.stop()

    n_baseline_force = max(1, min(len(force), int(round(0.2 * fs_force))))
    force_d = force - np.mean(force[:n_baseline_force])
    force_s = smooth_force(force_d, fs_force, cutoff=10.0)
    force_on_emg = np.interp(t_emg_s, t_force_s, force_s)

# =========================================================
# RESULTS
# =========================================================
results = {}

if do_time:
    results["EMG1_RMS_plateau"] = rms(x1[plateau_start:plateau_end])
    results["EMG1_IEMG_plateau"] = iemg(x1_rect[plateau_start:plateau_end], fs_emg)
    if onset_idx is not None:
        n200 = int(0.2 * fs_emg)
        results["EMG1_RMS_0_200ms"] = rms(x1[onset_idx:onset_idx+n200])

if do_freq:
    f1, pxx1 = welch_psd(x1, fs_emg)
    results["EMG1_MDF"] = mdf(f1, pxx1)
    results["EMG1_MPF"] = mpf(f1, pxx1)

if do_cci and x2_env is not None:
    results["CCI_plateau"] = cci(x1_env[plateau_start:plateau_end], x2_env[plateau_start:plateau_end])

if do_force and force_on_emg is not None:
    results["Force_peak"] = float(np.max(force_on_emg))
    results["Force_mean_plateau"] = float(np.mean(force_on_emg[plateau_start:plateau_end]))
    results["Force_CV_plateau"] = coeff_var(force_on_emg[plateau_start:plateau_end])
    if onset_idx is not None:
        results["Force_RFD_200ms"] = rfd(force_on_emg, fs_emg, onset_idx, window_ms=200)

# =========================================================
# PLOTS
# =========================================================
st.subheader("Grafici")
c1, c2 = st.columns(2)

with c1:
    fig = plt.figure()
    plt.plot(t_emg_s, emg1)
    plt.title("EMG1 raw")
    plt.xlabel("Tempo (s)")
    plt.ylabel("uV")
    st.pyplot(fig, clear_figure=True)

    fig = plt.figure()
    plt.plot(t_emg_s, x1)
    plt.title("EMG1 filtrato")
    plt.xlabel("Tempo (s)")
    plt.ylabel("uV")
    st.pyplot(fig, clear_figure=True)

with c2:
    fig = plt.figure()
    plt.plot(t_emg_s, x1_env)
    if onset_idx is not None:
        plt.axvline(t_emg_s[onset_idx], linestyle="--")
        plt.axhline(thr, linestyle="--")
    plt.title("Inviluppo EMG1")
    plt.xlabel("Tempo (s)")
    plt.ylabel("uV")
    st.pyplot(fig, clear_figure=True)

    if do_freq:
        fig = plt.figure()
        plt.semilogy(f1, pxx1)
        plt.title("PSD EMG1")
        plt.xlabel("Hz")
        plt.ylabel("Power")
        st.pyplot(fig, clear_figure=True)

if force_on_emg is not None:
    fig = plt.figure()
    plt.plot(t_emg_s, force_on_emg)
    plt.title("Forza interpolata su tempo EMG")
    plt.xlabel("Tempo (s)")
    plt.ylabel("N")
    st.pyplot(fig, clear_figure=True)

# =========================================================
# TABLE
# =========================================================
st.subheader("Risultati")
res_df = pd.DataFrame([results])
st.dataframe(res_df, use_container_width=True)

csv = res_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Scarica risultati CSV",
    data=csv,
    file_name="results_emg_force.csv",
    mime="text/csv"
)