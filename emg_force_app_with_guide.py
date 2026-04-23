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
from reportlab.lib.styles import ParagraphStyle

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="TitleCenter",
    parent=styles["Title"],
    alignment=1
))

styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading2"],
    spaceAfter=10
))

styles.add(ParagraphStyle(
    name="Body",
    parent=styles["Normal"],
    leading=14,
    spaceAfter=6
))

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

    def get_delta(name):
        if name not in summary_df.index:
            return np.nan
        val = summary_df.loc[name, "Delta_%"]
        return float(val) if np.isfinite(val) else np.nan

    def get_pre_post(name):
        if name not in summary_df.index:
            return np.nan, np.nan
        pre = summary_df.loc[name, "PRE_mean"]
        post = summary_df.loc[name, "POST_mean"]
        pre = float(pre) if np.isfinite(pre) else np.nan
        post = float(post) if np.isfinite(post) else np.nan
        return pre, post

    # Forza
    d_force_peak = get_delta("Force_peak")
    d_force_mvc = get_delta("Force_MVC_500ms")
    d_rfd = get_delta("Force_RFD_200ms")
    d_force_cv = get_delta("Force_CV")
    d_tremor = get_delta("Force_Tremor_3_7Hz")
    d_auc = get_delta("Force_AUC_above_50pct")

    # EMG tempo
    d_rms = get_delta("RMS")
    d_rms_norm = get_delta("RMS_norm")
    d_iemg = get_delta("IEMG")
    d_iemg_norm = get_delta("IEMG_norm")
    d_rms_200 = get_delta("RMS_0_200ms")

    # EMG frequenza / coordinazione
    d_mdf = get_delta("MDF")
    d_mpf = get_delta("MPF")
    d_cci = get_delta("CCI")

    # Efficienza
    d_eff = get_delta("Neuromuscular_Efficiency")

    # Timing
    d_onset = get_delta("Onset_s")

    # ===== Interpretazione forza =====
    if np.isfinite(d_force_peak):
        if d_force_peak > 10:
            text.append("✔ La forza di picco è aumentata in modo rilevante: il soggetto esprime una maggiore capacità massima.")
        elif d_force_peak < -10:
            text.append("⚠ La forza di picco è diminuita: possibile riduzione della capacità massima o variabilità tra trial.")

    if np.isfinite(d_force_mvc):
        if d_force_mvc > 10:
            text.append("✔ La MVC robusta su 500 ms è aumentata: il soggetto non solo raggiunge picchi più alti, ma mantiene anche livelli di forza più elevati.")
        elif d_force_mvc < -10:
            text.append("⚠ La MVC robusta su 500 ms è diminuita: la fase di forza sostenuta appare peggiore nel POST.")

    if np.isfinite(d_rfd):
        if d_rfd > 20:
            text.append("✔ L’RFD è aumentata nettamente: il sistema neuromuscolare sviluppa forza più rapidamente, segno di migliore esplosività.")
        elif d_rfd < -20:
            text.append("⚠ L’RFD è diminuita: la capacità di sviluppare forza rapidamente appare ridotta.")

    if np.isfinite(d_auc):
        if d_auc > 10:
            text.append("✔ L’area di forza sopra soglia è aumentata: il soggetto produce più forza complessiva nella parte alta della contrazione.")
        elif d_auc < -10:
            text.append("⚠ L’area di forza sopra soglia è diminuita: il contenuto complessivo della contrazione alta è inferiore.")

    # ===== Stabilità =====
    if np.isfinite(d_force_cv):
        if d_force_cv < -5:
            text.append("✔ Il coefficiente di variazione della forza è diminuito: la forza è più stabile.")
        elif d_force_cv > 5:
            text.append("⚠ Il coefficiente di variazione della forza è aumentato: la contrazione è più variabile. Questo però può anche accompagnare un aumento della forza espressa.")

    if np.isfinite(d_tremor):
        if d_tremor < -10:
            text.append("✔ La potenza del tremore 3–7 Hz è diminuita: possibile miglioramento della steadiness.")
        elif d_tremor > 10:
            text.append("⚠ La potenza del tremore 3–7 Hz è aumentata: la forza mostra più oscillazioni lente, potenzialmente legate al controllo neuromotorio o all’aumento del drive centrale.")

    # ===== EMG ampiezza =====
    if np.isfinite(d_rms_norm):
        if d_rms_norm > 10:
            text.append("✔ L’EMG normalizzato (RMS_norm) è aumentato: il muscolo lavora a una quota più alta del riferimento MVC EMG.")
        elif d_rms_norm < -10:
            text.append("✔ L’EMG normalizzato (RMS_norm) è diminuito: il muscolo ottiene il compito con minore attivazione relativa.")

    elif np.isfinite(d_rms):
        if d_rms > 10:
            text.append("✔ L’RMS è aumentato: maggiore attivazione neuromuscolare.")
        elif d_rms < -10:
            text.append("✔ L’RMS è diminuito: minore attivazione muscolare o migliore economia motoria.")

    if np.isfinite(d_iemg_norm):
        if d_iemg_norm > 10:
            text.append("✔ L’IEMG normalizzato è aumentato: il lavoro EMG totale è maggiore.")
        elif d_iemg_norm < -10:
            text.append("✔ L’IEMG normalizzato è diminuito: il lavoro EMG totale è minore.")

    elif np.isfinite(d_iemg):
        if d_iemg > 10:
            text.append("✔ L’IEMG è aumentato: il lavoro EMG totale è maggiore.")
        elif d_iemg < -10:
            text.append("✔ L’IEMG è diminuito: il lavoro EMG totale è minore.")

    if np.isfinite(d_rms_200):
        if d_rms_200 > 10:
            text.append("✔ L’RMS nei primi 200 ms è aumentato: migliore attivazione iniziale ed esplosività neurale.")
        elif d_rms_200 < -10:
            text.append("⚠ L’RMS nei primi 200 ms è diminuito: l’attivazione iniziale è meno rapida.")

    # ===== Frequenza / fatica =====
    if np.isfinite(d_mdf):
        if d_mdf > 5:
            text.append("✔ MDF in aumento: possibile riduzione della fatica o attivazione relativamente più veloce.")
        elif d_mdf < -5:
            text.append("⚠ MDF in diminuzione: possibile rallentamento della conduzione o comparsa di fatica.")

    if np.isfinite(d_mpf):
        if d_mpf > 5:
            text.append("✔ MPF in aumento: distribuzione spettrale spostata verso frequenze più alte.")
        elif d_mpf < -5:
            text.append("⚠ MPF in diminuzione: distribuzione spettrale spostata verso frequenze più basse.")

    if np.isfinite(d_cci):
        if d_cci < -5:
            text.append("✔ La co-contrazione è diminuita: possibile miglioramento dell’efficienza motoria.")
        elif d_cci > 5:
            text.append("⚠ La co-contrazione è aumentata: possibile maggiore rigidità o strategia di stabilizzazione.")

    # ===== Efficienza =====
    if np.isfinite(d_eff):
        if d_eff > 10:
            text.append("🔥 L’efficienza neuromuscolare è aumentata: il soggetto produce più forza per unità di attivazione EMG normalizzata.")
        elif d_eff < -10:
            text.append("⚠ L’efficienza neuromuscolare è diminuita: serve più attivazione relativa per produrre la forza osservata.")

    # ===== Timing =====
    if np.isfinite(d_onset):
        if d_onset < -5:
            text.append("✔ L’onset è anticipato: l’attivazione inizia prima.")
        elif d_onset > 5:
            text.append("⚠ L’onset è ritardato. Va però interpretato con cautela, perché dipende dal criterio di soglia e dalla qualità del segnale.")

    # ===== Sintesi finale =====
    positive_force = sum([
        np.isfinite(d_force_peak) and d_force_peak > 10,
        np.isfinite(d_force_mvc) and d_force_mvc > 10,
        np.isfinite(d_rfd) and d_rfd > 20
    ])

    positive_emg = sum([
        np.isfinite(d_rms_norm) and d_rms_norm > 10,
        np.isfinite(d_rms_200) and d_rms_200 > 10,
        np.isfinite(d_mdf) and d_mdf > 5,
        np.isfinite(d_mpf) and d_mpf > 5
    ])

    if positive_force >= 2 and positive_emg >= 1:
        text.append("🧠 Quadro complessivamente compatibile con un miglioramento neuromuscolare post-intervento, con aumento della capacità di forza e/o della qualità dell’attivazione.")
    elif positive_force >= 2:
        text.append("🧠 Quadro complessivamente compatibile con un miglioramento meccanico della prestazione di forza.")
    elif positive_emg >= 2:
        text.append("🧠 Quadro complessivamente compatibile con una modifica del pattern di attivazione neuromuscolare.")
    elif len(text) == 0:
        text.append("Nessuna variazione chiara o coerente rilevata nelle metriche principali.")

    return "\\n\\n".join(text)

def interpret_results_pdf(summary_df):
    sections = []

    def get_delta(name):
        if name not in summary_df.index:
            return np.nan
        val = summary_df.loc[name, "Delta_%"]
        return float(val) if np.isfinite(val) else np.nan

    d_force = get_delta("Force_peak")
    d_rfd = get_delta("Force_RFD_200ms")
    d_rms = get_delta("RMS_norm") if "RMS_norm" in summary_df.index else get_delta("RMS")
    d_cci = get_delta("CCI")
    d_mdf = get_delta("MDF")
    d_cv = get_delta("Force_CV")

    force_lines = []
    if np.isfinite(d_force):
        if d_force > 10:
            force_lines.append("✔ <b>Forza massima aumentata</b> → miglioramento della capacità neuromuscolare")
        elif d_force < -10:
            force_lines.append("⚠ <b>Forza massima diminuita</b>")

    if np.isfinite(d_rfd):
        if d_rfd > 20:
            force_lines.append("🔥 <b>RFD aumentata</b> → maggiore esplosività e reclutamento rapido")
        elif d_rfd < -20:
            force_lines.append("⚠ <b>RFD diminuita</b>")

    if force_lines:
        sections.append("<b>💪 FORZA</b><br/>" + "<br/>".join(force_lines))

    emg_lines = []
    if np.isfinite(d_rms):
        if d_rms > 10:
            emg_lines.append("✔ <b>Attivazione EMG aumentata</b> → maggiore drive neurale")
        elif d_rms < -10:
            emg_lines.append("✔ <b>Attivazione EMG ridotta</b> → possibile miglior efficienza")

    if np.isfinite(d_mdf):
        if d_mdf > 5:
            emg_lines.append("✔ <b>Frequenze EMG più alte</b> → minore fatica")
        elif d_mdf < -5:
            emg_lines.append("⚠ <b>Possibile fatica muscolare</b>")

    if emg_lines:
        sections.append("<b>⚡ EMG</b><br/>" + "<br/>".join(emg_lines))

    coord_lines = []
    if np.isfinite(d_cci):
        if d_cci < -5:
            coord_lines.append("✔ <b>Co-contrazione ridotta</b> → movimento più efficiente")
        elif d_cci > 5:
            coord_lines.append("⚠ <b>Co-contrazione aumentata</b>")

    if coord_lines:
        sections.append("<b>🤝 COORDINAZIONE</b><br/>" + "<br/>".join(coord_lines))

    stab_lines = []
    if np.isfinite(d_cv):
        if d_cv < -5:
            stab_lines.append("✔ <b>Forza più stabile</b>")
        elif d_cv > 5:
            stab_lines.append("⚠ <b>Maggiore variabilità della forza</b>")

    if stab_lines:
        sections.append("<b>🎯 STABILITÀ</b><br/>" + "<br/>".join(stab_lines))

    if not sections:
        return "Nessuna variazione significativa rilevata."

    return "<br/><br/>".join(sections)

from scipy.integrate import trapezoid

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

    return float(trapezoid(pxx[m], f[m]))


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

def interpret_results_dashboard(summary_df):
    def get_delta(name):
        if name not in summary_df.index:
            return np.nan
        val = summary_df.loc[name, "Delta_%"]
        return float(val) if np.isfinite(val) else np.nan

    sections = {
        "Forza": [],
        "Stabilità della forza": [],
        "EMG - attivazione": [],
        "EMG - frequenza": [],
        "Coordinazione": [],
        "Efficienza neuromuscolare": [],
        "Timing": [],
        "Sintesi finale": [],
    }

    d_force_peak = get_delta("Force_peak")
    d_force_mvc = get_delta("Force_MVC_500ms")
    d_force_mean = get_delta("Force_mean")
    d_rfd = get_delta("Force_RFD_200ms")
    d_force_cv = get_delta("Force_CV")
    d_tremor = get_delta("Force_Tremor_3_7Hz")

    d_rms = get_delta("RMS")
    d_rms_norm = get_delta("RMS_norm")
    d_iemg = get_delta("IEMG")
    d_iemg_norm = get_delta("IEMG_norm")
    d_rms_200 = get_delta("RMS_0_200ms")

    d_mdf = get_delta("MDF")
    d_mpf = get_delta("MPF")
    d_cci = get_delta("CCI")
    d_eff = get_delta("Neuromuscular_Efficiency")
    d_onset = get_delta("Onset_s")

    # FORZA
    if np.isfinite(d_force_peak):
        if d_force_peak > 10:
            sections["Forza"].append(
                f"La forza di picco è aumentata del {d_force_peak:.1f}%, suggerendo una maggiore capacità di esprimere forza massima."
            )
        elif d_force_peak < -10:
            sections["Forza"].append(
                f"La forza di picco è diminuita del {abs(d_force_peak):.1f}%, suggerendo una riduzione della capacità massima o maggiore variabilità tra i trial."
            )

    if np.isfinite(d_force_mvc):
        if d_force_mvc > 10:
            sections["Forza"].append(
                f"La MVC robusta su 500 ms è aumentata del {d_force_mvc:.1f}%, indicando una migliore capacità di sostenere livelli elevati di forza."
            )
        elif d_force_mvc < -10:
            sections["Forza"].append(
                f"La MVC robusta su 500 ms è diminuita del {abs(d_force_mvc):.1f}%, suggerendo una prestazione peggiore nella fase di forza mantenuta."
            )

    if np.isfinite(d_force_mean):
        if d_force_mean > 10:
            sections["Forza"].append(
                f"La forza media sopra soglia è aumentata del {d_force_mean:.1f}%, indicando una migliore qualità della contrazione nella parte alta del segnale."
            )
        elif d_force_mean < -10:
            sections["Forza"].append(
                f"La forza media sopra soglia è diminuita del {abs(d_force_mean):.1f}%, suggerendo un mantenimento meno efficace della forza."
            )

    if np.isfinite(d_rfd):
        if d_rfd > 20:
            sections["Forza"].append(
                f"L’RFD è aumentata del {d_rfd:.1f}%, segnalando una maggiore esplosività e una più rapida capacità di sviluppare forza."
            )
        elif d_rfd < -20:
            sections["Forza"].append(
                f"L’RFD è diminuita del {abs(d_rfd):.1f}%, suggerendo una minore esplosività neuromuscolare."
            )

    # STABILITÀ
    if np.isfinite(d_force_cv):
        if d_force_cv < -5:
            sections["Stabilità della forza"].append(
                f"Il coefficiente di variazione della forza è diminuito del {abs(d_force_cv):.1f}%, indicando una contrazione più stabile."
            )
        elif d_force_cv > 5:
            sections["Stabilità della forza"].append(
                f"Il coefficiente di variazione della forza è aumentato del {d_force_cv:.1f}%. La forza è quindi più variabile; questo dato va però interpretato insieme all’eventuale aumento della forza espressa."
            )

    if np.isfinite(d_tremor):
        if d_tremor < -10:
            sections["Stabilità della forza"].append(
                f"La potenza del tremore 3–7 Hz è diminuita del {abs(d_tremor):.1f}%, suggerendo un possibile miglioramento della steadiness."
            )
        elif d_tremor > 10:
            sections["Stabilità della forza"].append(
                f"La potenza del tremore 3–7 Hz è aumentata del {d_tremor:.1f}%, indicando oscillazioni più evidenti della forza."
            )

    # EMG ATTIVAZIONE
    if np.isfinite(d_rms_norm):
        if d_rms_norm > 10:
            sections["EMG - attivazione"].append(
                f"L’RMS normalizzato è aumentato del {d_rms_norm:.1f}%, indicando che il muscolo lavora a una quota più alta del riferimento MVC EMG."
            )
        elif d_rms_norm < -10:
            sections["EMG - attivazione"].append(
                f"L’RMS normalizzato è diminuito del {abs(d_rms_norm):.1f}%, suggerendo una minore attivazione relativa o una migliore economia neuromuscolare."
            )
    elif np.isfinite(d_rms):
        if d_rms > 10:
            sections["EMG - attivazione"].append(
                f"L’RMS è aumentato del {d_rms:.1f}%, compatibilmente con una maggiore attivazione muscolare."
            )
        elif d_rms < -10:
            sections["EMG - attivazione"].append(
                f"L’RMS è diminuito del {abs(d_rms):.1f}%, indicando una minore ampiezza del segnale EMG."
            )

    if np.isfinite(d_iemg_norm):
        if d_iemg_norm > 10:
            sections["EMG - attivazione"].append(
                f"L’IEMG normalizzato è aumentato del {d_iemg_norm:.1f}%, suggerendo un maggiore lavoro EMG totale."
            )
        elif d_iemg_norm < -10:
            sections["EMG - attivazione"].append(
                f"L’IEMG normalizzato è diminuito del {abs(d_iemg_norm):.1f}%, indicando un lavoro EMG totale minore."
            )
    elif np.isfinite(d_iemg):
        if d_iemg > 10:
            sections["EMG - attivazione"].append(
                f"L’IEMG è aumentato del {d_iemg:.1f}%, confermando un incremento dell’attività muscolare complessiva."
            )
        elif d_iemg < -10:
            sections["EMG - attivazione"].append(
                f"L’IEMG è diminuito del {abs(d_iemg):.1f}%, indicando una riduzione dell’attività muscolare complessiva."
            )

    if np.isfinite(d_rms_200):
        if d_rms_200 > 10:
            sections["EMG - attivazione"].append(
                f"L’RMS nei primi 200 ms è aumentato del {d_rms_200:.1f}%, suggerendo una fase iniziale di attivazione più rapida ed esplosiva."
            )
        elif d_rms_200 < -10:
            sections["EMG - attivazione"].append(
                f"L’RMS nei primi 200 ms è diminuito del {abs(d_rms_200):.1f}%, suggerendo un reclutamento iniziale meno rapido."
            )

    # EMG FREQUENZA
    if np.isfinite(d_mdf):
        if d_mdf > 5:
            sections["EMG - frequenza"].append(
                f"La MDF è aumentata del {d_mdf:.1f}%, suggerendo uno spostamento dello spettro verso frequenze più alte e una possibile minore fatica."
            )
        elif d_mdf < -5:
            sections["EMG - frequenza"].append(
                f"La MDF è diminuita del {abs(d_mdf):.1f}%, dato compatibile con affaticamento o rallentamento della conduzione."
            )

    if np.isfinite(d_mpf):
        if d_mpf > 5:
            sections["EMG - frequenza"].append(
                f"La MPF è aumentata del {d_mpf:.1f}%, confermando uno spostamento del contenuto spettrale verso frequenze più alte."
            )
        elif d_mpf < -5:
            sections["EMG - frequenza"].append(
                f"La MPF è diminuita del {abs(d_mpf):.1f}%, indicando uno spostamento verso frequenze più basse."
            )

    # COORDINAZIONE
    if np.isfinite(d_cci):
        if d_cci < -5:
            sections["Coordinazione"].append(
                f"La co-contrazione è diminuita del {abs(d_cci):.1f}%, suggerendo un pattern motorio più efficiente."
            )
        elif d_cci > 5:
            sections["Coordinazione"].append(
                f"La co-contrazione è aumentata del {d_cci:.1f}%, possibile segno di maggiore rigidità o stabilizzazione."
            )

    # EFFICIENZA
    if np.isfinite(d_eff):
        if d_eff > 10:
            sections["Efficienza neuromuscolare"].append(
                f"L’efficienza neuromuscolare è aumentata del {d_eff:.1f}%, indicando una maggiore forza prodotta per unità di attivazione EMG normalizzata."
            )
        elif d_eff < -10:
            sections["Efficienza neuromuscolare"].append(
                f"L’efficienza neuromuscolare è diminuita del {abs(d_eff):.1f}%, indicando un costo neurale relativo più elevato per produrre la forza osservata."
            )

    # TIMING
    if np.isfinite(d_onset):
        if d_onset < -5:
            sections["Timing"].append(
                f"L’onset è anticipato del {abs(d_onset):.1f}%, quindi l’attivazione EMG compare prima."
            )
        elif d_onset > 5:
            sections["Timing"].append(
                f"L’onset è ritardato del {d_onset:.1f}%. Questo dato va interpretato con cautela perché dipende dalla soglia di rilevamento."
            )

    # SINTESI
    miglioramento_forza = (
        (np.isfinite(d_force_peak) and d_force_peak > 10) or
        (np.isfinite(d_force_mvc) and d_force_mvc > 10) or
        (np.isfinite(d_rfd) and d_rfd > 20)
    )

    miglioramento_emg = (
        (np.isfinite(d_rms_norm) and d_rms_norm > 10) or
        (np.isfinite(d_rms) and d_rms > 10) or
        (np.isfinite(d_rms_200) and d_rms_200 > 10)
    )

    miglioramento_eff = np.isfinite(d_eff) and d_eff > 10

    if miglioramento_forza and miglioramento_emg:
        sections["Sintesi finale"].append(
            "Nel complesso il quadro è compatibile con un miglioramento neuromuscolare post-intervento, con incremento della capacità di forza e un pattern di attivazione più favorevole."
        )
    elif miglioramento_forza:
        sections["Sintesi finale"].append(
            "Nel complesso emerge soprattutto un miglioramento meccanico della prestazione di forza."
        )
    elif miglioramento_emg:
        sections["Sintesi finale"].append(
            "Nel complesso emerge soprattutto una modifica del pattern di attivazione neuromuscolare."
        )
    else:
        sections["Sintesi finale"].append(
            "Non emergono variazioni univoche e fortemente coerenti su tutte le metriche principali; il risultato va interpretato con cautela."
        )

    if miglioramento_eff:
        sections["Sintesi finale"].append(
            "L’aumento dell’efficienza neuromuscolare suggerisce che il sistema produce più forza a fronte di un costo neurale relativo più favorevole."
        )

    return sections

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


import html
import re

def markdown_to_reportlab_html(text: str) -> str:
    """
    Converte un markdown molto semplice in HTML compatibile con ReportLab.
    Supporta:
    - ### Titoli
    - **grassetto**
    - newline -> <br/>
    """
    lines = text.splitlines()
    out_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            out_lines.append("<br/>")
            continue

        # Titoli markdown ### -> <b>...</b>
        if line.startswith("### "):
            line = f"<b>{html.escape(line[4:])}</b>"
        else:
            line = html.escape(line)

        # **bold** -> <b>...</b>
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

        out_lines.append(line)

    return "<br/>".join(out_lines)

def create_zip_report(trial_df, summary_df):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trial_results.csv", trial_df.to_csv(index=False))
        zf.writestr("summary_pre_post.csv", summary_df.to_csv())
    bio.seek(0)
    return bio

def show_interpretation_dashboard(summary_df):
    sections = interpret_results_dashboard(summary_df)

    st.subheader("🧠 Interpretazione automatica")

    order = [
        "Forza",
        "Stabilità della forza",
        "EMG - attivazione",
        "EMG - frequenza",
        "Coordinazione",
        "Efficienza neuromuscolare",
        "Timing",
        "Sintesi finale",
    ]

    shown_any = False

    for title in order:
        lines = sections.get(title, [])
        if not lines:
            continue

        shown_any = True
        with st.container(border=True):
            st.markdown(f"### {title}")
            for line in lines:
                st.write(f"- {line}")

    if not shown_any:
        st.info("Nessuna variazione significativa rilevata.")


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

def interpret_results_pdf(summary_df):
    parts = []

    def get_delta(name):
        if name not in summary_df.index:
            return np.nan
        val = summary_df.loc[name, "Delta_%"]
        return float(val) if np.isfinite(val) else np.nan

    def has_metric(name):
        return name in summary_df.index and np.isfinite(summary_df.loc[name, "POST_mean"])

    d_force_peak = get_delta("Force_peak")
    d_force_mvc = get_delta("Force_MVC_500ms")
    d_force_mean = get_delta("Force_mean")
    d_rfd = get_delta("Force_RFD_200ms")
    d_force_cv = get_delta("Force_CV")
    d_tremor = get_delta("Force_Tremor_3_7Hz")

    d_rms = get_delta("RMS")
    d_rms_norm = get_delta("RMS_norm")
    d_iemg = get_delta("IEMG")
    d_iemg_norm = get_delta("IEMG_norm")
    d_rms_200 = get_delta("RMS_0_200ms")

    d_mdf = get_delta("MDF")
    d_mpf = get_delta("MPF")
    d_cci = get_delta("CCI")
    d_eff = get_delta("Neuromuscular_Efficiency")
    d_onset = get_delta("Onset_s")

    # =========================
    # FORZA
    # =========================
    forza = []
    if np.isfinite(d_force_peak):
        if d_force_peak > 10:
            forza.append(
                "La forza di picco è aumentata in modo rilevante, suggerendo una maggiore capacità del soggetto di esprimere forza massima."
            )
        elif d_force_peak < -10:
            forza.append(
                "La forza di picco è diminuita, suggerendo una riduzione della capacità massima oppure una maggiore variabilità tra i trial."
            )

    if np.isfinite(d_force_mvc):
        if d_force_mvc > 10:
            forza.append(
                "La MVC robusta calcolata su finestra mobile di 500 ms è aumentata, indicando che il soggetto non solo raggiunge picchi più alti, ma riesce anche a sostenere livelli elevati di forza in maniera più stabile."
            )
        elif d_force_mvc < -10:
            forza.append(
                "La MVC robusta su 500 ms è diminuita, suggerendo una prestazione peggiore nella fase di forza sostenuta."
            )

    if np.isfinite(d_force_mean):
        if d_force_mean > 10:
            forza.append(
                "La forza media sopra soglia è aumentata, indicando una migliore capacità di mantenere la contrazione in una zona funzionalmente rilevante."
            )
        elif d_force_mean < -10:
            forza.append(
                "La forza media sopra soglia è diminuita, suggerendo una riduzione della qualità della contrazione mantenuta."
            )

    if np.isfinite(d_rfd):
        if d_rfd > 20:
            forza.append(
                "L’RFD è aumentata in maniera marcata, suggerendo una migliore capacità di sviluppare forza rapidamente. Questo risultato è particolarmente rilevante perché riflette l’efficienza della fase iniziale di reclutamento neuromuscolare."
            )
        elif d_rfd < -20:
            forza.append(
                "L’RFD è diminuita, suggerendo una minore esplosività neuromuscolare nella fase iniziale della contrazione."
            )

    if forza:
        parts.append("<b>💪 Forza</b><br/>" + "<br/>".join([f"• {x}" for x in forza]))

    # =========================
    # STABILITÀ
    # =========================
    stabilita = []
    if np.isfinite(d_force_cv):
        if d_force_cv < -5:
            stabilita.append(
                "Il coefficiente di variazione della forza è diminuito: la contrazione appare più stabile e con minori fluttuazioni relative."
            )
        elif d_force_cv > 5:
            stabilita.append(
                "Il coefficiente di variazione della forza è aumentato: la contrazione appare più variabile. Questo dato va però interpretato insieme all’aumento della forza, perché una forza più alta può accompagnarsi a una maggiore variabilità assoluta."
            )

    if np.isfinite(d_tremor):
        if d_tremor < -10:
            stabilita.append(
                "La potenza del tremore nella banda 3–7 Hz è diminuita, suggerendo un possibile miglioramento della steadiness e del controllo fine della forza."
            )
        elif d_tremor > 10:
            stabilita.append(
                "La potenza del tremore nella banda 3–7 Hz è aumentata, indicando oscillazioni più evidenti nel segnale di forza. Questo può riflettere un aumento del drive centrale, ma anche una minore regolarità del controllo."
            )

    if stabilita:
        parts.append("<b>🎯 Stabilità della forza</b><br/>" + "<br/>".join([f"• {x}" for x in stabilita]))

    # =========================
    # EMG AMPIEZZA
    # =========================
    emg_amp = []
    if np.isfinite(d_rms_norm):
        if d_rms_norm > 10:
            emg_amp.append(
                "L’RMS normalizzato è aumentato: il muscolo lavora a una quota più alta del riferimento MVC EMG, suggerendo un aumento dell’attivazione relativa."
            )
        elif d_rms_norm < -10:
            emg_amp.append(
                "L’RMS normalizzato è diminuito: il soggetto ottiene la prestazione con minore attivazione relativa, possibile segno di maggiore economia neuromuscolare."
            )
    elif np.isfinite(d_rms):
        if d_rms > 10:
            emg_amp.append(
                "L’RMS è aumentato: il segnale EMG mostra una maggiore ampiezza, compatibile con un incremento del drive neurale e/o del reclutamento di unità motorie."
            )
        elif d_rms < -10:
            emg_amp.append(
                "L’RMS è diminuito: il segnale EMG mostra una minore ampiezza, interpretabile come minore attivazione o migliore efficienza a parità di compito."
            )

    if np.isfinite(d_iemg_norm):
        if d_iemg_norm > 10:
            emg_amp.append(
                "L’IEMG normalizzato è aumentato: il lavoro EMG totale nella finestra analizzata è maggiore."
            )
        elif d_iemg_norm < -10:
            emg_amp.append(
                "L’IEMG normalizzato è diminuito: il lavoro EMG totale appare ridotto."
            )
    elif np.isfinite(d_iemg):
        if d_iemg > 10:
            emg_amp.append(
                "L’IEMG è aumentato, indicando un maggiore contenuto complessivo di attività EMG nel tempo."
            )
        elif d_iemg < -10:
            emg_amp.append(
                "L’IEMG è diminuito, indicando un minore contenuto complessivo di attività EMG nel tempo."
            )

    if np.isfinite(d_rms_200):
        if d_rms_200 > 10:
            emg_amp.append(
                "L’RMS nei primi 200 ms è aumentato, suggerendo una fase iniziale di attivazione più rapida ed esplosiva."
            )
        elif d_rms_200 < -10:
            emg_amp.append(
                "L’RMS nei primi 200 ms è diminuito, suggerendo una fase iniziale di reclutamento meno rapida."
            )

    if emg_amp:
        parts.append("<b>⚡ EMG – ampiezza e attivazione</b><br/>" + "<br/>".join([f"• {x}" for x in emg_amp]))

    # =========================
    # EMG FREQUENZA
    # =========================
    emg_freq = []
    if np.isfinite(d_mdf):
        if d_mdf > 5:
            emg_freq.append(
                "La MDF è aumentata: il contenuto spettrale dell’EMG si è spostato verso frequenze più alte, dato compatibile con minore fatica o attivazione relativamente più rapida."
            )
        elif d_mdf < -5:
            emg_freq.append(
                "La MDF è diminuita: possibile rallentamento della conduzione o presenza di fatica muscolare."
            )

    if np.isfinite(d_mpf):
        if d_mpf > 5:
            emg_freq.append(
                "La MPF è aumentata, confermando uno spostamento dello spettro verso frequenze più alte."
            )
        elif d_mpf < -5:
            emg_freq.append(
                "La MPF è diminuita, suggerendo uno spostamento dello spettro verso frequenze più basse."
            )

    if emg_freq:
        parts.append("<b>📊 EMG – frequenza</b><br/>" + "<br/>".join([f"• {x}" for x in emg_freq]))

    # =========================
    # COORDINAZIONE
    # =========================
    coord = []
    if np.isfinite(d_cci):
        if d_cci < -5:
            coord.append(
                "La co-contrazione è diminuita: questo può riflettere un pattern motorio più efficiente, con minore attivazione antagonista non necessaria."
            )
        elif d_cci > 5:
            coord.append(
                "La co-contrazione è aumentata: questo può riflettere una maggiore rigidità articolare o una strategia di stabilizzazione."
            )

    if coord:
        parts.append("<b>🤝 Coordinazione</b><br/>" + "<br/>".join([f"• {x}" for x in coord]))

    # =========================
    # EFFICIENZA
    # =========================
    eff = []
    if np.isfinite(d_eff):
        if d_eff > 10:
            eff.append(
                "L’efficienza neuromuscolare è aumentata: il soggetto produce più forza per unità di attivazione EMG normalizzata."
            )
        elif d_eff < -10:
            eff.append(
                "L’efficienza neuromuscolare è diminuita: per produrre la forza osservata è richiesta una maggiore attivazione relativa."
            )

    if eff:
        parts.append("<b>🧠 Efficienza neuromuscolare</b><br/>" + "<br/>".join([f"• {x}" for x in eff]))

    # =========================
    # TIMING
    # =========================
    timing = []
    if np.isfinite(d_onset):
        if d_onset < -5:
            timing.append(
                "L’onset EMG è anticipato: l’attivazione inizia prima rispetto alla condizione PRE."
            )
        elif d_onset > 5:
            timing.append(
                "L’onset EMG è ritardato. Questo dato va interpretato con cautela, perché dipende dal criterio di soglia, dalla baseline e dalla qualità del segnale."
            )

    if timing:
        parts.append("<b>⏱ Timing</b><br/>" + "<br/>".join([f"• {x}" for x in timing]))

    # =========================
    # SINTESI FINALE
    # =========================
    sintesi = []

    miglioramento_forza = (
        (np.isfinite(d_force_peak) and d_force_peak > 10) or
        (np.isfinite(d_force_mvc) and d_force_mvc > 10) or
        (np.isfinite(d_rfd) and d_rfd > 20)
    )

    miglioramento_attivazione = (
        (np.isfinite(d_rms_norm) and d_rms_norm > 10) or
        (np.isfinite(d_rms) and d_rms > 10) or
        (np.isfinite(d_rms_200) and d_rms_200 > 10)
    )

    miglioramento_efficienza = np.isfinite(d_eff) and d_eff > 10

    if miglioramento_forza and miglioramento_attivazione:
        sintesi.append(
            "Nel complesso, il quadro è compatibile con un miglioramento neuromuscolare post-intervento, caratterizzato da aumento della forza e da una modifica favorevole dell’attivazione EMG."
        )
    elif miglioramento_forza:
        sintesi.append(
            "Nel complesso, il quadro mostra soprattutto un miglioramento meccanico della prestazione di forza."
        )
    elif miglioramento_attivazione:
        sintesi.append(
            "Nel complesso, il quadro mostra soprattutto una modifica del pattern di attivazione neuromuscolare."
        )

    if miglioramento_efficienza:
        sintesi.append(
            "L’aumento dell’efficienza neuromuscolare suggerisce che il sistema produce più forza a fronte di un costo neurale relativo più favorevole."
        )

    if not sintesi:
        sintesi.append(
            "Non emergono variazioni univoche e fortemente coerenti in tutte le metriche principali; il risultato va quindi interpretato con cautela e integrato con l’ispezione dei singoli trial."
        )

    parts.append("<b>📌 Sintesi conclusiva</b><br/>" + "<br/>".join([f"• {x}" for x in sintesi]))

    return "<br/><br/>".join(parts)

def create_pdf_report(summary_df, pre_details=None, post_details=None):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    from reportlab.lib.styles import ParagraphStyle

    styles.add(ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=1,
        spaceAfter=14
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        spaceBefore=10,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=6
    ))

    elements = []

    # Titolo
    elements.append(Paragraph("Report EMG + Forza PRE vs POST", styles["TitleCenter"]))
    elements.append(Spacer(1, 8))

    # =========================
    # Summary table
    # =========================
    elements.append(Paragraph("Risultati sintetici", styles["SectionTitle"]))

    df_print = summary_df.reset_index().copy()
    df_print = df_print.round(3)
    df_print.columns = ["Parametro", "PRE", "POST", "Delta", "Delta %"]
    df_print["Parametro"] = df_print["Parametro"].astype(str).str.replace("_", " ", regex=False)

    data = [df_print.columns.tolist()] + df_print.values.tolist()

    table = Table(data, colWidths=[150, 70, 70, 70, 70])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    # =========================
    # Guida breve
    # =========================
    elements.append(Paragraph("Guida sintetica alle metriche", styles["SectionTitle"]))

    protocollo_pdf = """
<b>Protocollo sperimentale</b><br/><br/>

<b>Obiettivo dello studio</b><br/>
Valutare se l’esposizione a vibrazione segmentale ad alta frequenza produce modificazioni della performance neuromuscolare osservabili nel segnale EMG e nella forza durante una contrazione volontaria massimale (MVC).<br/><br/>

<b>Setup sperimentale</b><br/>
Il soggetto esegue una presa massimale su una maniglia strumentata con due sensori estensimetrici:
- Push: faccia anteriore della maniglia
- Pull: faccia posteriore della maniglia<br/><br/>

Durante la prova vengono registrati:
- segnale di forza
- segnale EMG del canale 1
- segnale EMG del canale 2<br/><br/>

<b>Sequenza delle misure</b><br/>
1. Fase PRE: tre prove di MVC della durata di alcuni secondi<br/>
2. Intervento vibrazionale: tre esposizioni da 10 minuti a 100 Hz<br/>
3. Fase POST: ripetizione delle prove di MVC con lo stesso setup<br/><br/>

<b>Cosa viene confrontato</b><br/>
L’analisi confronta PRE e POST per valutare modifiche di:
- forza massima
- rapidità di sviluppo della forza
- attivazione EMG
- co-contrazione
- stabilità della forza
- efficienza neuromuscolare<br/><br/>

<b>Ipotesi fisiologica</b><br/>
La vibrazione segmentale può indurre facilitazione neuromuscolare, con possibile aumento del reclutamento, della sincronizzazione e della prestazione meccanica.<br/><br/>
"""

    elements.append(Paragraph("Protocollo sperimentale", styles["SectionTitle"]))
    elements.append(Paragraph(protocollo_pdf, styles["BodyTextCustom"]))
    elements.append(Spacer(1, 12))
    guida = """
    <b>EMG – ampiezza</b><br/>
    RMS = ampiezza efficace del segnale EMG; IEMG = contenuto totale di attività nel tempo; RMS 0–200 ms = attivazione iniziale rapida.<br/><br/>

    <b>EMG – frequenza</b><br/>
    MDF e MPF descrivono la distribuzione spettrale del segnale e sono utili per interpretare fatica e qualità dell’attivazione.<br/><br/>

    <b>Coordinazione</b><br/>
    CCI descrive la co-contrazione tra due muscoli, utile per valutare rigidità o efficienza del pattern motorio.<br/><br/>

    <b>Forza</b><br/>
    Force peak = massimo istantaneo; Force MVC 500 ms = massima media su finestra mobile; Force CV = variabilità relativa; Force RFD = rapidità di sviluppo della forza; Force AUC = quantità totale di forza sopra soglia.<br/><br/>

    <b>Normalizzazione EMG</b><br/>
    RMS_norm e IEMG_norm esprimono il segnale EMG in rapporto al riferimento MVC EMG, rendendo più confrontabili i trial.<br/><br/>

    <b>Efficienza neuromuscolare</b><br/>
    Rapporto tra forza prodotta e attivazione EMG normalizzata.
    """

    elements.append(Paragraph(guida, styles["BodyTextCustom"]))
    elements.append(Spacer(1, 12))

    # =========================
    # Interpretazione automatica
    # =========================
    elements.append(Paragraph("Interpretazione automatica", styles["SectionTitle"]))
    interp_html = interpret_results_pdf(summary_df)
    elements.append(Paragraph(interp_html, styles["BodyTextCustom"]))
    elements.append(Spacer(1, 14))

    # =========================
    # Grafici
    # =========================
    if pre_details and post_details and len(pre_details) > 0 and len(post_details) > 0:
        pre0 = pre_details[0]
        post0 = post_details[0]

        elements.append(Paragraph("Grafici rappresentativi", styles["SectionTitle"]))

        # EMG filtrato
        fig = plt.figure(figsize=(7, 3))
        plt.plot(pre0["t_s"], pre0["emg_filt"], label="PRE")
        plt.plot(post0["t_s"], post0["emg_filt"], label="POST")
        plt.title("EMG filtrato PRE vs POST")
        plt.xlabel("Tempo (s)")
        plt.ylabel("uV")
        plt.legend()
        elements.append(Paragraph("Figura 1. EMG filtrato", styles["BodyTextCustom"]))
        elements.append(fig_to_image(fig, width=500, height=210))
        elements.append(Spacer(1, 10))

        # Inviluppo
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
        elements.append(Paragraph("Figura 2. Inviluppo EMG", styles["BodyTextCustom"]))
        elements.append(fig_to_image(fig, width=500, height=210))
        elements.append(Spacer(1, 10))

        # PSD
        fig = plt.figure(figsize=(7, 3))
        if len(pre0["f_psd"]) > 0 and len(pre0["p_psd"]) > 0:
            plt.semilogy(pre0["f_psd"], pre0["p_psd"], label="PRE")
        if len(post0["f_psd"]) > 0 and len(post0["p_psd"]) > 0:
            plt.semilogy(post0["f_psd"], post0["p_psd"], label="POST")
        plt.title("PSD PRE vs POST")
        plt.xlabel("Hz")
        plt.ylabel("Power")
        plt.legend()
        elements.append(Paragraph("Figura 3. Spettro di potenza EMG", styles["BodyTextCustom"]))
        elements.append(fig_to_image(fig, width=500, height=210))
        elements.append(Spacer(1, 10))

        # Forza
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
            elements.append(Paragraph("Figura 4. Traccia di forza", styles["BodyTextCustom"]))
            elements.append(fig_to_image(fig, width=500, height=210))
            elements.append(Spacer(1, 10))

    # =========================
    # Bibliografia
    # =========================
    elements.append(Paragraph("Riferimenti bibliografici essenziali", styles["SectionTitle"]))

    refs = """
    1. De Luca CJ. The use of surface electromyography in biomechanics.<br/>
    2. Burden A. How should we normalize electromyograms obtained from healthy participants?<br/>
    3. Maffiuletti NA et al. Rate of force development: physiological and methodological considerations.<br/>
    4. Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy.<br/>
    5. McAuley JH, Marsden CD. Physiological and pathological tremors and rhythmic central motor control.
    """

    elements.append(Paragraph(refs, styles["BodyTextCustom"]))

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

PROTOCOLLO_STUDIO = """
# 📋 Protocollo sperimentale

## Obiettivo dello studio
L’obiettivo dell’analisi è valutare se l’esposizione a vibrazione segmentale ad alta frequenza produce modificazioni nella performance neuromuscolare del soggetto, osservabili sia nel segnale elettromiografico (EMG) sia nella forza espressa durante una contrazione volontaria massimale (MVC).

In particolare, vogliamo capire se dopo l’intervento si osservano:

- aumento della forza massima
- aumento della rapidità di sviluppo della forza
- aumento o riorganizzazione dell’attivazione EMG
- riduzione della co-contrazione non necessaria
- miglioramento della stabilità della forza
- miglioramento dell’efficienza neuromuscolare

---

## Setup sperimentale
Il soggetto esegue una presa massimale su una maniglia strumentata.

La maniglia è dotata di due sensori estensimetrici:
- **Push**: sensore sulla faccia anteriore della maniglia
- **Pull**: sensore sulla faccia posteriore della maniglia

Il segnale di forza può essere costruito a partire dai due canali in modi diversi. Nel protocollo di presa, la scelta più rappresentativa della forza totale di grip è generalmente la somma:

$$
F_{grip}(t) = Push(t) + Pull(t)
$$

Durante la prova, due elettrodi di superficie registrano il segnale EMG su due muscoli, per esempio flessore ed estensore dell’avambraccio.

---

## Sequenza delle misure
Il protocollo è strutturato in due fasi principali:

### 1. Fase PRE
Il soggetto esegue tre prove di contrazione volontaria massimale (MVC) della durata di alcuni secondi.

Durante ogni prova vengono registrati:
- segnale di forza
- segnale EMG del canale 1
- segnale EMG del canale 2

### 2. Intervento vibrazionale
Il soggetto viene sottoposto a tre sessioni di esposizione a vibrazione segmentale mediante dispositivo strumentale.

Parametri principali dell’intervento:
- durata di ciascuna esposizione: **10 minuti**
- frequenza vibrazione: **100 Hz**
- ampiezza: dell’ordine di alcuni micrometri

### 3. Fase POST
Dopo le esposizioni a vibrazione, il soggetto ripete le prove di MVC con lo stesso setup sperimentale.

Anche in questa fase vengono registrati:
- segnale di forza
- segnale EMG del canale 1
- segnale EMG del canale 2

---

## Cosa confrontiamo
L’analisi confronta la condizione **PRE** e la condizione **POST**.

Per ogni trial vengono calcolate metriche relative a:

### EMG
- ampiezza del segnale
- lavoro EMG totale
- contenuto spettrale
- co-contrazione
- attivazione iniziale

### Forza
- forza massima
- forza media sostenuta
- stabilità della forza
- esplosività
- tremore
- area sopra soglia

### Integrazione EMG–Forza
- efficienza neuromuscolare
- correlazione tra attivazione e output meccanico

---

## Ipotesi fisiologica
L’ipotesi di lavoro è che la vibrazione segmentale possa modificare il comportamento neuromuscolare attraverso:

- facilitazione del drive afferente
- aumento del reclutamento delle unità motorie
- miglioramento della sincronizzazione
- cambiamento del pattern di attivazione agonista/antagonista
- aumento dell’efficienza del sistema neuromuscolare

Di conseguenza, nel POST potremmo osservare:
- incremento della forza
- aumento dell’RFD
- aumento di RMS e/o RMS normalizzato
- modifiche favorevoli di MDF e MPF
- riduzione della CCI
- migliore steadiness della forza

---

## Logica dell’analisi
Per ridurre la variabilità e rendere i confronti più robusti:

- i trial vengono analizzati separatamente
- si calcolano medie PRE e POST
- molte metriche di forza vengono valutate solo nella porzione del segnale sopra una soglia relativa (es. 50% del picco)
- l’EMG può essere normalizzato su un riferimento MVC EMG
- l’interpretazione finale integra insieme forza, EMG, coordinazione e stabilità

---
"""

st.markdown(PROTOCOLLO_STUDIO)

GUIDA_COMPLETA = """
# 🧠 1. STRUTTURA GENERALE

- **PRE_mean** → media dei trial prima
- **POST_mean** → media dei trial dopo
- **Delta** → differenza assoluta
- **Delta_%** → variazione percentuale

👉 Serve a confrontare in modo aggregato l’effetto dell’intervento tra condizioni.

---

# ⚡ 2. EMG – DOMINIO DEL TEMPO

## 🔹 RMS (Root Mean Square)

**Formula**
$$
RMS = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} x_i^2}
$$

**A cosa serve**
👉 Misura l’ampiezza efficace del segnale EMG.  
👉 È uno degli indicatori più usati dell’attivazione neuromuscolare.

**Significato fisiologico**
- aumenta quando cresce il reclutamento delle unità motorie
- aumenta quando cresce il drive neurale complessivo

**Interpretazione**
- ↑ RMS → maggiore attivazione muscolare
- ↓ RMS → minore attivazione o possibile fatica / inefficienza

**Riferimenti**
- De Luca, 1997
- Farina et al., 2004

---

## 🔹 IEMG (Integrated EMG)

**Formula**
$$
IEMG = \\frac{\\sum |x(t)|}{f_s}
$$

**A cosa serve**
👉 Misura il contenuto totale di attività EMG nel tempo.  
👉 È utile per quantificare il “lavoro elettrico” totale del muscolo nella finestra analizzata.

**Significato fisiologico**
- combina ampiezza e durata della contrazione
- è sensibile all’attivazione totale sviluppata nel tempo

**Interpretazione**
- ↑ IEMG → maggiore lavoro muscolare totale
- ↓ IEMG → minore lavoro o finestra meno attiva

**Riferimenti**
- De Luca, 1997
- Merletti & Parker, 2004

---

## 🔹 RMS 0–200 ms

**A cosa serve**
👉 Misura l’attivazione EMG nella fase iniziale della contrazione.  
👉 È molto utile per valutare la componente esplosiva del reclutamento.

**Significato fisiologico**
- riflette la rapidità di attivazione neurale
- si collega bene alla capacità di sviluppare forza rapidamente

**Interpretazione**
- ↑ RMS 0–200 ms → migliore attivazione iniziale
- ↓ RMS 0–200 ms → risposta iniziale meno esplosiva

**Riferimenti**
- Aagaard et al., 2002
- Maffiuletti et al., 2016

---

# 📊 3. EMG – DOMINIO DELLA FREQUENZA

## 🔹 MDF (Median Frequency)

**Formula**
$$
\\int_0^{MDF} PSD(f)\\,df = \\frac{1}{2}\\int_0^{f_{max}} PSD(f)\\,df
$$

**A cosa serve**
👉 Valuta come è distribuita la potenza del segnale EMG nello spettro.  
👉 È molto usata negli studi di fatica.

**Significato fisiologico**
- legata alla velocità di conduzione delle fibre
- influenzata dallo stato di fatica e dal tipo di reclutamento

**Interpretazione**
- ↓ MDF → possibile fatica o rallentamento della conduzione
- ↑ MDF → attivazione relativamente più “veloce”

**Riferimenti**
- De Luca, 1997

---

## 🔹 MPF (Mean Power Frequency)

**Formula**
$$
MPF = \\frac{\\sum f \\cdot PSD(f)}{\\sum PSD(f)}
$$

**A cosa serve**
👉 Fornisce il “baricentro” dello spettro di potenza EMG.  
👉 È utile per valutare modifiche della distribuzione spettrale.

**Significato fisiologico**
- riflette velocità di conduzione, tipo di fibre e stato di fatica
- è spesso interpretata insieme a MDF

**Interpretazione**
- ↓ MPF → possibile fatica / rallentamento
- ↑ MPF → attivazione relativamente più rapida

**Riferimenti**
- De Luca, 1997

---

# 🤝 4. CO-CONTRAZIONE

## 🔹 CCI (Co-Contraction Index)

**Formula**
$$
CCI = \\frac{2 \\cdot min(A,B)}{A+B}
$$

**A cosa serve**
👉 Misura quanto due muscoli agonista/antagonista lavorano insieme.  
👉 Utile per valutare stabilizzazione articolare ed efficienza motoria.

**Significato fisiologico**
- valori alti indicano maggiore co-attivazione
- può riflettere rigidità, stabilizzazione o strategia compensatoria

**Interpretazione**
- ↑ CCI → maggiore co-contrazione
- ↓ CCI → minore co-contrazione, spesso maggiore efficienza

**Riferimenti**
- De Luca, 1997

---

# 💪 5. FORZA – METRICHE PRINCIPALI

## 🔹 Force_peak

**Formula**
$$
F_{peak} = max(F(t))
$$

**A cosa serve**
👉 È il massimo valore istantaneo di forza.  
👉 Rappresenta il picco meccanico massimo osservato.

**Interpretazione**
- ↑ Force_peak → maggiore forza massima espressa

**Riferimenti**
- Maffiuletti et al., 2016

---

## 🔹 Force_MVC_500ms

**Definizione**
Massima media di forza su una finestra mobile di 500 ms.

**A cosa serve**
👉 È una stima più robusta della MVC rispetto al singolo picco.  
👉 Riduce l’influenza di spike e artefatti.

**Interpretazione**
- ↑ → migliore capacità di sostenere una forza elevata

**Riferimenti**
- approccio coerente con la letteratura sulle MVC isometriche e con le raccomandazioni metodologiche sull’RFD

---

## 🔹 Force_top5_mean

**Definizione**
Media del 5% dei campioni di forza più alti.

**A cosa serve**
👉 Fornisce una misura robusta della parte alta del segnale, meno sensibile al singolo outlier.

---

## 🔹 Force_mean_above_50pct

**Definizione**
Media della forza per i campioni sopra il 50% del picco.

**A cosa serve**
👉 Quantifica la forza nel tratto “utile” e confrontabile della contrazione.

---

## 🔹 Force_CV / Force_CV_above_50pct

**Formula**
$$
CV = \\frac{\\sigma}{\\mu}
$$

**A cosa serve**
👉 Misura la stabilità della forza.  
👉 È uno degli indici più usati per la force steadiness.

**Interpretazione**
- ↑ CV → maggiore variabilità / minore stabilità
- ↓ CV → maggiore stabilità

**Riferimenti**
- Enoka & Duchateau, 2008

---

## 🔹 Force_AUC_above_50pct

**Formula**
$$
AUC = \\sum F(t) \\cdot \\Delta t
$$

**A cosa serve**
👉 Misura la quantità totale di forza sviluppata sopra soglia.  
👉 È utile se vuoi quantificare il “contenuto” complessivo della contrazione.

---

# 🔬 6. STABILITÀ DELLA FORZA

## 🔹 Force_Tremor_3_7Hz

**Formula**
Integrazione della PSD della forza tra 3 e 7 Hz.

**A cosa serve**
👉 Quantifica le oscillazioni lente della forza in una banda legata al tremore fisiologico / al controllo motorio.

**Interpretazione**
- ↑ tremor power → maggiori oscillazioni / possibile minore steadiness
- ↓ tremor power → contrazione più stabile

**Riferimenti**
- McAuley & Marsden, 2000

---

## 🔹 Force_SampEn

**Formula**
$$
SampEn = -\\ln(A/B)
$$

**A cosa serve**
👉 Misura l’irregolarità / complessità del segnale di forza.  
👉 È utile per descrivere il controllo motorio oltre alla sola variabilità.

**Interpretazione**
- valori più alti → maggiore complessità / minore regolarità
- valori più bassi → segnale più prevedibile / più rigido

**Riferimenti**
- Richman & Moorman, 2000

---

## 🔹 Force_Stability_Index

**Definizione**
Indice composito costruito combinando variabilità, tremore e complessità.

**A cosa serve**
👉 Riassume in un singolo numero la qualità di stabilità della forza.

**Nota**
È un indice composito personalizzato: utile, ma va descritto esplicitamente nella sezione Metodi.

---

# 🚀 7. ESPLOSIVITÀ

## 🔹 Force_RFD_200ms

**Formula**
$$
RFD = \\frac{\\Delta F}{\\Delta t}
$$

**A cosa serve**
👉 Misura quanto rapidamente il soggetto sviluppa forza.  
👉 È una metrica chiave dell’esplosività neuromuscolare.

**Interpretazione**
- ↑ RFD → migliore capacità esplosiva

**Riferimenti**
- Maffiuletti et al., 2016

---

# 🔗 8. NORMALIZZAZIONE EMG

## 🔹 RMS_norm / IEMG_norm

**Formula**
$$
EMG_{norm} = \\frac{EMG}{EMG_{MVC}} \\times 100
$$

**A cosa serve**
👉 Rende confrontabili trial e sessioni diverse.  
👉 Riduce l’effetto di fattori strumentali, posizionamento elettrodi e impedenza cutanea.

**Interpretazione**
- 100% = livello EMG di riferimento MVC
- >100% = possibile superamento del riferimento scelto o riferimento PRE basso

**Riferimenti**
- Burden, 2010

---

# 🧠 9. EFFICIENZA NEUROMUSCOLARE

## 🔹 Neuromuscular_Efficiency

**Formula**
$$
Efficiency = \\frac{Force\\_MVC\\_500ms}{RMS_{norm}}
$$

**A cosa serve**
👉 Stima quanta forza viene prodotta per unità di attivazione EMG normalizzata.  
👉 È utile per capire se il sistema neuromuscolare sta diventando più “economico”.

**Interpretazione**
- ↑ efficienza → più forza con minore costo neurale relativo
- ↓ efficienza → maggiore attivazione per produrre la stessa forza

**Nota**
Il concetto di rapporto EMG–forza è consolidato; il nome specifico dell’indice va sempre definito chiaramente nei Metodi.

**Riferimenti**
- De Luca, 1997
- Burden, 2010

---

# 📚 10. RIFERIMENTI BIBLIOGRAFICI ESSENZIALI

1. De Luca CJ. The use of surface electromyography in biomechanics.
2. Burden A. How should we normalize electromyograms obtained from healthy participants?
3. Maffiuletti NA et al. Rate of force development: physiological and methodological considerations.
4. Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy.
5. McAuley JH, Marsden CD. Physiological and pathological tremors and rhythmic central motor control.

---

👉 Questa pipeline consente di analizzare attivazione, frequenza, coordinazione, stabilità, esplosività ed efficienza neuromuscolare in modo integrato.
"""


st.markdown(GUIDA_COMPLETA)

st.subheader("Interpretazione automatica")
show_interpretation_dashboard(summary_df)

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