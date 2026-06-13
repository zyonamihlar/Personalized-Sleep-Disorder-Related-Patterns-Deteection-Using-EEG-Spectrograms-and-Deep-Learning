import os
import shutil
import tempfile
import threading
from datetime import datetime

from tkinter import filedialog, messagebox

import numpy as np
import torch
import torch.nn as nn
import mne
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler

import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    MODEL_DIR,
    NUM_CLASSES,
    CLASS_NAMES,
    LOW_FREQ,
    HIGH_FREQ,
    TARGET_SAMPLE_RATE,
    EEG_CHANNEL,
    EPOCH_SECONDS,
    N_PER_SEG,
    N_OVERLAP
)


# ============================================================
# Global settings
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selected_file = ""
last_report_text = ""

# Keep the latest results so charts can be re-rendered on a theme switch
last_results = {
    "epoch0": None,
    "sfreq": None,
    "preds": None,
    "summary": None,
}

spectrogram_canvas = None
stage_chart_canvas = None
hypnogram_canvas = None


# ============================================================
# CNN model architecture
# Must match train_model.py and personalize_model.py
# ============================================================
class SimpleSleepCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Model loading
# ============================================================
def load_model(preference="Personalized"):
    """
    preference: "Personalized" (default) or "Generic".
    If the requested model is missing, fall back to the other one.
    """
    personalized_path = os.path.join(MODEL_DIR, "personalized_sleep_cnn.pth")
    generic_path = os.path.join(MODEL_DIR, "generic_sleep_cnn.pth")

    model = SimpleSleepCNN().to(DEVICE)

    # Ordered list of (path, name) to try, based on the user's preference
    if preference == "Generic":
        candidates = [(generic_path, "Generic CNN"), (personalized_path, "Personalized CNN")]
    else:
        candidates = [(personalized_path, "Personalized CNN"), (generic_path, "Generic CNN")]

    for path, name in candidates:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            return model, name

    raise FileNotFoundError(
        "No trained model was found.\n\n"
        "Please run:\n"
        "1. train_model.py\n"
        "2. personalize_model.py"
    )


# ============================================================
# File handling
# ============================================================
def prepare_signal_file(file_path):
    """
    MNE reads EDF files directly.
    ISRUC REC files are copied temporarily as EDF because MNE checks file extension.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".edf":
        return file_path, None

    if extension == ".rec":
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        temp_file.close()

        shutil.copy(file_path, temp_file.name)
        return temp_file.name, temp_file.name

    raise ValueError("Unsupported file type. Please select an EDF or REC file.")


def select_best_eeg_channel(raw):
    """
    Supports both Sleep-EDF and ISRUC-style EEG channel names.
    """

    possible_channels = [
        EEG_CHANNEL,
        "EEG Fpz-Cz",
        "Fpz-Cz",
        "F3-A2",
        "C3-A2",
        "C4-A1",
        "F4-A1",
        "O1-A2",
        "O2-A1"
    ]

    for channel in possible_channels:
        if channel in raw.ch_names:
            return channel

    raise ValueError(
        "No compatible EEG channel was found.\n\n"
        f"Available channels:\n{raw.ch_names}"
    )


# ============================================================
# EEG preprocessing
# ============================================================
def preprocess_uploaded_signal(file_path):
    usable_path, temp_path = prepare_signal_file(file_path)

    try:
        raw = mne.io.read_raw_edf(usable_path, preload=True, verbose=False)

        selected_channel = select_best_eeg_channel(raw)

        raw.pick([selected_channel])
        raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
        raw.resample(TARGET_SAMPLE_RATE, verbose=False)

        signal = raw.get_data()[0]
        sfreq = int(raw.info["sfreq"])
        epoch_samples = sfreq * EPOCH_SECONDS

        epochs = []

        for start in range(0, len(signal) - epoch_samples + 1, epoch_samples):
            end = start + epoch_samples
            epochs.append(signal[start:end])

        if len(epochs) == 0:
            raise ValueError("No valid 30-second epochs could be extracted.")

        return np.array(epochs), sfreq, selected_channel

    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# Spectrogram generation
# ============================================================
def make_spectrogram(epoch_signal, sfreq):
    freqs, times, spec = spectrogram(
        epoch_signal,
        fs=sfreq,
        nperseg=N_PER_SEG,
        noverlap=N_OVERLAP
    )

    mask = freqs <= 30
    spec = spec[mask, :]

    spec = np.log1p(spec)

    scaler = MinMaxScaler()
    spec = scaler.fit_transform(spec)

    return spec.astype(np.float32)


# ============================================================
# Prediction
# ============================================================
def predict_sleep_stages(model, epochs, sfreq):
    spectrograms = []

    for epoch in epochs:
        spec = make_spectrogram(epoch, sfreq)
        spectrograms.append(spec)

    X = np.array(spectrograms)

    if len(X) == 0:
        raise ValueError("No spectrograms were generated.")

    X = X[:, np.newaxis, :, :]
    X_tensor = torch.tensor(X, dtype=torch.float32)

    predictions = []

    batch_size = 128

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start:start + batch_size].to(DEVICE)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    return np.array(predictions)


# ============================================================
# Summary and disorder indicators
# ============================================================
def summarize_predictions(preds):
    total = len(preds)

    summary = {}

    for i, stage in enumerate(CLASS_NAMES):
        count = int(np.sum(preds == i))
        percentage = (count / total) * 100 if total > 0 else 0
        summary[stage] = {
            "count": count,
            "percentage": percentage
        }

    wake_ratio = np.sum(preds == 0) / total if total > 0 else 0
    n3_ratio = np.sum(preds == 3) / total if total > 0 else 0
    rem_ratio = np.sum(preds == 4) / total if total > 0 else 0

    insomnia_flag = wake_ratio > 0.35
    fragmented_flag = (rem_ratio < 0.10) or (n3_ratio < 0.10)

    return summary, {
        "wake_ratio": wake_ratio,
        "n3_ratio": n3_ratio,
        "rem_ratio": rem_ratio,
        "insomnia_flag": insomnia_flag,
        "fragmented_flag": fragmented_flag
    }


# ============================================================
# Report generation
# ============================================================
def build_report(file_path, model_name, channel, sfreq, epoch_count, summary, indicators):
    report = []
    report.append("PERSONALIZED EEG SLEEP ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Input file: {file_path}")
    report.append(f"Model used: {model_name}")
    report.append(f"Selected EEG channel: {channel}")
    report.append(f"Sampling frequency after preprocessing: {sfreq} Hz")
    report.append(f"Total 30-second epochs analysed: {epoch_count}")
    report.append("")
    report.append("SLEEP STAGE DISTRIBUTION")
    report.append("-" * 50)

    for stage in CLASS_NAMES:
        count = summary[stage]["count"]
        percentage = summary[stage]["percentage"]
        report.append(f"{stage:<8}: {count:>6} epochs   ({percentage:6.2f}%)")

    report.append("")
    report.append("PRELIMINARY DISORDER-RELATED INDICATORS")
    report.append("-" * 50)
    report.append(f"Wake ratio: {indicators['wake_ratio']:.4f}")
    report.append(f"N3 ratio  : {indicators['n3_ratio']:.4f}")
    report.append(f"REM ratio : {indicators['rem_ratio']:.4f}")
    report.append(
        f"Possible insomnia-related pattern: "
        f"{'Yes' if indicators['insomnia_flag'] else 'No'}"
    )
    report.append(
        f"Possible fragmented sleep pattern: "
        f"{'Yes' if indicators['fragmented_flag'] else 'No'}"
    )

    report.append("")
    report.append("NOTE")
    report.append("-" * 50)
    report.append(
        "These outputs represent preliminary pattern indicators generated from "
        "EEG-based sleep-stage predictions. They are not intended to replace "
        "clinical diagnosis or expert sleep-laboratory assessment."
    )

    return "\n".join(report)


# ============================================================
# Theme / palette
# ============================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

ACCENT = "#6366f1"        # indigo-500
ACCENT_HOVER = "#4f46e5"  # indigo-600
TEAL = "#14b8a6"
TEAL_HOVER = "#0d9488"
SLATE = "#475569"
SLATE_HOVER = "#334155"

GOOD = "#22c55e"
WARN = "#f59e0b"
INFO = "#6366f1"

# Per-stage colours (light -> deep through the sleep cycle)
STAGE_COLORS = {
    "Wake": "#f59e0b",   # amber
    "N1": "#38bdf8",     # sky
    "N2": "#3b82f6",     # blue
    "N3": "#6366f1",     # indigo
    "REM": "#a855f7",    # purple
}

FONT_FAMILY = "Segoe UI"


def chart_colors():
    """Return matplotlib colours matched to the current appearance mode."""
    if ctk.get_appearance_mode() == "Dark":
        return {
            "card": "#1d1e26",
            "text": "#e6e8ee",
            "muted": "#9aa3b2",
            "grid": "#2c2e3a",
        }
    return {
        "card": "#ffffff",
        "text": "#1f2937",
        "muted": "#64748b",
        "grid": "#e5e7eb",
    }


def style_axes(fig, ax, c):
    fig.patch.set_facecolor(c["card"])
    ax.set_facecolor(c["card"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(c["text"])
    ax.yaxis.label.set_color(c["text"])
    ax.title.set_color(c["text"])


# ============================================================
# Plotting
# ============================================================
def clear_canvas(canvas):
    if canvas is not None:
        canvas.get_tk_widget().destroy()


def _mount(fig, frame):
    c = chart_colors()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(bg=c["card"], highlightthickness=0)
    widget.pack(fill="both", expand=True, padx=6, pady=6)
    plt.close(fig)
    return canvas


def plot_spectrogram(epoch_signal, sfreq):
    global spectrogram_canvas
    clear_canvas(spectrogram_canvas)
    c = chart_colors()

    freqs, times, spec = spectrogram(
        epoch_signal, fs=sfreq, nperseg=N_PER_SEG, noverlap=N_OVERLAP
    )
    mask = freqs <= 30
    freqs = freqs[mask]
    spec = np.log1p(spec[mask, :])

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=100)
    image = ax.pcolormesh(times, freqs, spec, shading="gouraud", cmap="magma")

    style_axes(fig, ax, c)
    ax.set_title("EEG Spectrogram  ·  Epoch 1", fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=c["muted"], length=0)
    cbar.set_label("Log Power", color=c["text"])

    fig.tight_layout()
    spectrogram_canvas = _mount(fig, spectrogram_frame)


def plot_stage_distribution(summary):
    global stage_chart_canvas
    clear_canvas(stage_chart_canvas)
    c = chart_colors()

    stages = list(summary.keys())
    percentages = [summary[s]["percentage"] for s in stages]
    colors = [STAGE_COLORS.get(s, ACCENT) for s in stages]

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=100)
    bars = ax.bar(stages, percentages, color=colors, width=0.62, zorder=3)

    style_axes(fig, ax, c)
    ax.set_title("Sleep Stage Distribution", fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, max(percentages) + 12 if max(percentages) > 0 else 10)
    ax.yaxis.grid(True, color=c["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    for rect, value in zip(bars, percentages):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 1.5,
                f"{value:.1f}%", ha="center", fontsize=9,
                color=c["text"], fontweight="bold")

    fig.tight_layout()
    stage_chart_canvas = _mount(fig, stage_chart_frame)


def plot_hypnogram(preds):
    global hypnogram_canvas
    clear_canvas(hypnogram_canvas)
    c = chart_colors()

    epoch_numbers = np.arange(len(preds))

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=100)
    ax.plot(epoch_numbers, preds, drawstyle="steps-post",
            linewidth=1.4, color=ACCENT, zorder=3)
    ax.fill_between(epoch_numbers, preds, step="post",
                    alpha=0.12, color=ACCENT, zorder=2)

    style_axes(fig, ax, c)
    ax.set_title("Predicted Sleep Stage Timeline", fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Sleep Stage")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.grid(True, color=c["grid"], alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    hypnogram_canvas = _mount(fig, hypnogram_frame)


def refresh_plots():
    """Re-render charts after a theme switch, if results exist."""
    if last_results["preds"] is None:
        return
    plot_spectrogram(last_results["epoch0"], last_results["sfreq"])
    plot_stage_distribution(last_results["summary"])
    plot_hypnogram(last_results["preds"])


# ============================================================
# UI helpers
# ============================================================
def card(parent, **kwargs):
    return ctk.CTkFrame(parent, corner_radius=16, fg_color=("#ffffff", "#1d1e26"),
                        border_width=1, border_color=("#e5e7eb", "#2a2c38"), **kwargs)


def heading(parent, text, **pack):
    lbl = ctk.CTkLabel(parent, text=text, font=(FONT_FAMILY, 15, "bold"),
                       anchor="w")
    lbl.pack(fill="x", padx=18, pady=(16, 8), **pack)
    return lbl


def thread_safe(fn):
    """Run a UI update on the main thread from any thread."""
    root.after(0, fn)


# ============================================================
# State updates
# ============================================================
def set_status(text):
    thread_safe(lambda: status_label.configure(text=text))


def set_badge(text, color):
    def _apply():
        status_chip.configure(text="  " + text + "  ", fg_color=color)
    thread_safe(_apply)


def log(message):
    def _apply():
        log_box.configure(state="normal")
        log_box.insert("end", message)
        log_box.see("end")
        log_box.configure(state="disabled")
    thread_safe(_apply)


def set_busy(busy):
    def _apply():
        state = "disabled" if busy else "normal"

        for b in (select_btn, run_btn, export_btn):
            b.configure(state=state)

        if busy:
            progress.pack(fill="x", padx=18, pady=(0, 12))
            progress.configure(mode="indeterminate")
            progress.start()
        else:
            progress.stop()
            progress.pack_forget()

    thread_safe(_apply)


def animate_bar(bar, target, value_label, count, current=0.0):
    """Smoothly grow a progress bar up to its target fraction."""
    step = 0.04
    current = min(current + step, target)
    bar.set(current)
    value_label.configure(text=f"{count}  ·  {target * 100:.1f}%")
    if current < target - 1e-9:
        root.after(12, lambda: animate_bar(bar, target, value_label, count, current))


def choose_file():
    global selected_file
    file_path = filedialog.askopenfilename(
        title="Select EEG File",
        filetypes=[
            ("EEG files", "*.edf *.rec"),
            ("EDF files", "*.edf"),
            ("REC files", "*.rec"),
            ("All files", "*.*"),
        ],
    )
    if file_path:
        selected_file = file_path
        file_name_var.configure(text=os.path.basename(file_path))
        file_path_var.configure(text=file_path)
        set_badge("File selected", INFO)
        set_status("EEG file selected. Ready to run analysis.")


# ============================================================
# Analysis (runs off the main thread so the UI stays responsive)
# ============================================================
def run_analysis():
    if not selected_file:
        messagebox.showwarning("No file selected", "Please select an EDF or REC file first.")
        return

    set_busy(True)
    log_box.configure(state="normal")
    log_box.delete("1.0", "end")
    log_box.configure(state="disabled")
    set_badge("Working…", INFO)

    threading.Thread(target=_analysis_worker, daemon=True).start()


def _analysis_worker():
    global last_report_text
    try:
        preference = model_choice.get()
        set_status("Loading trained model...")
        log(f"Loading {preference.lower()} model...\n")
        model, model_name = load_model(preference)
        log(f"Model loaded: {model_name}\n")

        set_status("Preprocessing EEG signal...")
        log("Reading and preprocessing EEG signal...\n")
        epochs, sfreq, selected_channel = preprocess_uploaded_signal(selected_file)
        log(f"Selected EEG channel: {selected_channel}\n")
        log(f"Sampling rate after preprocessing: {sfreq} Hz\n")
        log(f"Extracted epochs: {len(epochs)}\n")

        set_status("Running CNN sleep-stage prediction...")
        log("Generating spectrograms and running CNN predictions...\n")
        preds = predict_sleep_stages(model, epochs, sfreq)

        summary, indicators = summarize_predictions(preds)

        last_report_text = build_report(
            selected_file, model_name, selected_channel, sfreq,
            len(epochs), summary, indicators,
        )

        last_results.update({
            "epoch0": epochs[0], "sfreq": sfreq,
            "preds": preds, "summary": summary,
        })

        thread_safe(lambda: _render_results(
            model_name, selected_channel, sfreq, len(epochs),
            summary, indicators, epochs[0], preds, last_report_text,
        ))


    except Exception as error:
        error_message = str(error)
        thread_safe(lambda: _fail(error_message))


def _render_results(model_name, channel, sfreq, epoch_count,
                    summary, indicators, epoch0, preds, report_text):
    global placeholder
    # Remove the empty-state placeholder once we have real charts
    if placeholder is not None:
        placeholder.destroy()
        placeholder = None

    # Metric cards
    metric_values["Model"].configure(text=model_name)
    metric_values["Channel"].configure(text=channel)
    metric_values["Sampling"].configure(text=f"{sfreq} Hz")
    metric_values["Epochs"].configure(text=str(epoch_count))

    # Animated stage bars
    for stage in CLASS_NAMES:
        info = summary[stage]
        animate_bar(stage_bars[stage], info["percentage"] / 100.0,
                    stage_values[stage], info["count"])

    # Indicators
    efficiency = (1 - indicators["wake_ratio"]) * 100
    indicator_values["Sleep Efficiency"].configure(text=f"{efficiency:.1f}%")
    indicator_values["Wake Ratio"].configure(text=f"{indicators['wake_ratio']:.2f}")
    indicator_values["N3 Ratio"].configure(text=f"{indicators['n3_ratio']:.2f}")
    indicator_values["REM Ratio"].configure(text=f"{indicators['rem_ratio']:.2f}")

    if indicators["insomnia_flag"]:
        insomnia_chip.configure(text="  Detected  ", fg_color=WARN)
    else:
        insomnia_chip.configure(text="  Normal  ", fg_color=GOOD)
    if indicators["fragmented_flag"]:
        fragmented_chip.configure(text="  Detected  ", fg_color=WARN)
    else:
        fragmented_chip.configure(text="  Normal  ", fg_color=GOOD)

    # Charts
    plot_spectrogram(epoch0, sfreq)
    plot_stage_distribution(summary)
    plot_hypnogram(preds)

    log("\nAnalysis completed successfully.\n\n")
    log(report_text)

    set_badge("Analysis complete", GOOD)
    set_status("Analysis completed successfully.")
    set_busy(False)


def _fail(error):
    set_badge("Analysis failed", WARN)
    set_status("Analysis failed.")
    set_busy(False)
    messagebox.showerror("Analysis Error", str(error))


def export_report():
    if not last_report_text:
        messagebox.showwarning("No report available",
                               "Please run an analysis before exporting a report.")
        return
    save_path = filedialog.asksaveasfilename(
        title="Save Analysis Report", defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
    )
    if save_path:
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(last_report_text)
        messagebox.showinfo("Report exported", f"Report saved to:\n{save_path}")


def on_appearance_change(choice):
    ctk.set_appearance_mode(choice)
    root.after(120, refresh_plots)


# ============================================================
# Layout
# ============================================================
root = ctk.CTk()
root.title("Personalized EEG Sleep Analysis System")
root.geometry("1500x920")
root.minsize(1280, 820)

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)


# ---- Header ------------------------------------------------------------------
header = ctk.CTkFrame(root, corner_radius=0, height=78, fg_color=("#0f172a", "#0b0d14"))
header.grid(row=0, column=0, sticky="ew")
header.grid_propagate(False)
header.grid_columnconfigure(1, weight=1)

logo = ctk.CTkLabel(header, text="🌙", font=(FONT_FAMILY, 26),
                    fg_color=ACCENT, corner_radius=12, width=46, height=46)
logo.grid(row=0, column=0, padx=(22, 14), pady=16)

title_box = ctk.CTkFrame(header, fg_color="transparent")
title_box.grid(row=0, column=1, sticky="w")
ctk.CTkLabel(title_box, text="Personalized EEG Sleep Analysis System",
             font=(FONT_FAMILY, 20, "bold"), text_color="#ffffff").pack(anchor="w")
ctk.CTkLabel(title_box,
             text="Spectrogram-based CNN sleep staging with preliminary disorder indication",
             font=(FONT_FAMILY, 12), text_color="#94a3b8").pack(anchor="w")

appearance = ctk.CTkSegmentedButton(header, values=["Light", "Dark", "System"],
                                    command=on_appearance_change)
appearance.set("Dark")
appearance.grid(row=0, column=2, padx=22)


# ---- Body --------------------------------------------------------------------
body = ctk.CTkFrame(root, fg_color="transparent")
body.grid(row=1, column=0, sticky="nsew", padx=18, pady=18)
body.grid_rowconfigure(0, weight=1)
body.grid_columnconfigure(1, weight=1)

# ---- Sidebar -----------------------------------------------------------------
sidebar = ctk.CTkFrame(body, width=270, corner_radius=16,
                       fg_color=("#ffffff", "#1d1e26"),
                       border_width=1, border_color=("#e5e7eb", "#2a2c38"))
sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
sidebar.grid_propagate(False)

heading(sidebar, "Input Control")
ctk.CTkLabel(sidebar, text="Select an EEG recording and run the trained model.",
             font=(FONT_FAMILY, 12), text_color=("#64748b", "#9aa3b2"),
             wraplength=220, justify="left").pack(fill="x", padx=18, pady=(0, 14))

ctk.CTkLabel(sidebar, text="MODEL", font=(FONT_FAMILY, 10, "bold"),
             text_color=("#94a3b8", "#6b7280"), anchor="w").pack(fill="x", padx=18, pady=(0, 4))
model_choice = ctk.CTkOptionMenu(
    sidebar, values=["Personalized", "Generic"],
    height=38, corner_radius=10, font=(FONT_FAMILY, 13, "bold"),
    fg_color=("#eef0f4", "#262833"), button_color=ACCENT, button_hover_color=ACCENT_HOVER,
    text_color=("#1f2937", "#e6e8ee"))
model_choice.set("Personalized")  # personalized stays the default
model_choice.pack(fill="x", padx=18, pady=(0, 14))

select_btn = ctk.CTkButton(sidebar, text="📂   Select EEG File", command=choose_file,
                           height=44, corner_radius=12, font=(FONT_FAMILY, 13, "bold"),
                           fg_color=ACCENT, hover_color=ACCENT_HOVER)
select_btn.pack(fill="x", padx=18, pady=(0, 10))

run_btn = ctk.CTkButton(sidebar, text="▶   Run Analysis", command=run_analysis,
                        height=44, corner_radius=12, font=(FONT_FAMILY, 13, "bold"),
                        fg_color=TEAL, hover_color=TEAL_HOVER)
run_btn.pack(fill="x", padx=18, pady=(0, 10))

export_btn = ctk.CTkButton(sidebar, text="⬇   Export Report", command=export_report,
                           height=44, corner_radius=12, font=(FONT_FAMILY, 13, "bold"),
                           fg_color=SLATE, hover_color=SLATE_HOVER)
export_btn.pack(fill="x", padx=18, pady=(0, 12))

progress = ctk.CTkProgressBar(sidebar, height=8, corner_radius=4, progress_color=ACCENT)
progress.pack(fill="x", padx=18, pady=(0, 12))
progress.set(0)
progress.pack_forget()  # hidden until busy

status_chip = ctk.CTkLabel(sidebar, text="  Ready  ", font=(FONT_FAMILY, 12, "bold"),
                           fg_color=INFO, text_color="#ffffff", corner_radius=10)
status_chip.pack(anchor="w", padx=18, pady=(0, 18))

ctk.CTkFrame(sidebar, height=1, fg_color=("#e5e7eb", "#2a2c38")).pack(fill="x", padx=18)

heading(sidebar, "Selected File")
file_name_var = ctk.CTkLabel(sidebar, text="No file selected",
                             font=(FONT_FAMILY, 13, "bold"), anchor="w",
                             wraplength=220, justify="left")
file_name_var.pack(fill="x", padx=18, pady=(0, 2))
file_path_var = ctk.CTkLabel(sidebar, text="Choose an EDF or REC recording to begin.",
                             font=(FONT_FAMILY, 10), text_color=("#94a3b8", "#6b7280"),
                             anchor="w", wraplength=220, justify="left")
file_path_var.pack(fill="x", padx=18, pady=(0, 16))

ctk.CTkLabel(sidebar, text=f"Compute device:  {DEVICE.type.upper()}",
             font=(FONT_FAMILY, 11), text_color=("#94a3b8", "#6b7280"),
             anchor="w").pack(side="bottom", fill="x", padx=18, pady=16)


# ---- Center column -----------------------------------------------------------
center = ctk.CTkFrame(body, fg_color="transparent")
center.grid(row=0, column=1, sticky="nsew", padx=(0, 16))
center.grid_rowconfigure(1, weight=1)
center.grid_columnconfigure(0, weight=1)

# Metric cards row
metrics = ctk.CTkFrame(center, fg_color="transparent")
metrics.grid(row=0, column=0, sticky="ew", pady=(0, 16))
metric_values = {}
metric_specs = [("Model", "—"), ("Channel", "—"), ("Sampling", "—"), ("Epochs", "—")]
for i, (label, default) in enumerate(metric_specs):
    metrics.grid_columnconfigure(i, weight=1)
    mc = card(metrics)
    mc.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
    ctk.CTkLabel(mc, text=label.upper(), font=(FONT_FAMILY, 10, "bold"),
                 text_color=("#94a3b8", "#6b7280")).pack(anchor="w", padx=16, pady=(14, 0))
    val = ctk.CTkLabel(mc, text=default, font=(FONT_FAMILY, 18, "bold"))
    val.pack(anchor="w", padx=16, pady=(0, 14))
    metric_values[label] = val

# Visualisation card with tabs
viz = card(center)
viz.grid(row=1, column=0, sticky="nsew")
heading(viz, "Signal & Prediction Visualisation")

tabs = ctk.CTkTabview(viz, corner_radius=12, fg_color=("#f8fafc", "#15161d"),
                      segmented_button_selected_color=ACCENT,
                      segmented_button_selected_hover_color=ACCENT_HOVER)
tabs.pack(fill="both", expand=True, padx=14, pady=(0, 16))
tabs.add("Spectrogram")
tabs.add("Stage Distribution")
tabs.add("Prediction Timeline")

spectrogram_frame = tabs.tab("Spectrogram")
stage_chart_frame = tabs.tab("Stage Distribution")
hypnogram_frame = tabs.tab("Prediction Timeline")

placeholder = ctk.CTkLabel(
    spectrogram_frame,
    text="📈\n\nSelect an EEG file and run the analysis\nto see your results visualised here.",
    font=(FONT_FAMILY, 14), text_color=("#94a3b8", "#6b7280"), justify="center")
placeholder.pack(expand=True)


# ---- Right column (scrollable so it never gets clipped) ----------------------
right = ctk.CTkScrollableFrame(body, width=380, fg_color="transparent")
right.grid(row=0, column=2, sticky="nsew")
right.grid_columnconfigure(0, weight=1)

# Sleep stage results
stage_card = card(right)
stage_card.pack(fill="x", pady=(0, 16))
heading(stage_card, "Sleep Stage Results")

stage_bars = {}
stage_values = {}
for stage in CLASS_NAMES:
    row = ctk.CTkFrame(stage_card, fg_color="transparent")
    row.pack(fill="x", padx=18, pady=(2, 8))
    top = ctk.CTkFrame(row, fg_color="transparent")
    top.pack(fill="x")
    dot = ctk.CTkLabel(top, text="●", text_color=STAGE_COLORS[stage],
                       font=(FONT_FAMILY, 13))
    dot.pack(side="left")
    ctk.CTkLabel(top, text=f"  {stage}", font=(FONT_FAMILY, 13, "bold")).pack(side="left")
    val = ctk.CTkLabel(top, text="—", font=(FONT_FAMILY, 11),
                       text_color=("#64748b", "#9aa3b2"))
    val.pack(side="right")
    bar = ctk.CTkProgressBar(row, height=8, corner_radius=4,
                             progress_color=STAGE_COLORS[stage])
    bar.set(0)
    bar.pack(fill="x", pady=(6, 0))
    stage_bars[stage] = bar
    stage_values[stage] = val
ctk.CTkFrame(stage_card, height=6, fg_color="transparent").pack()

# Indicators
indicator_card = card(right)
indicator_card.pack(fill="x", pady=(0, 16))
heading(indicator_card, "Preliminary Indicators")

indicator_values = {}
for label in ("Sleep Efficiency", "Wake Ratio", "N3 Ratio", "REM Ratio"):
    r = ctk.CTkFrame(indicator_card, fg_color="transparent")
    r.pack(fill="x", padx=18, pady=4)
    ctk.CTkLabel(r, text=label, font=(FONT_FAMILY, 12),
                 text_color=("#64748b", "#9aa3b2")).pack(side="left")
    v = ctk.CTkLabel(r, text="—", font=(FONT_FAMILY, 13, "bold"))
    v.pack(side="right")
    indicator_values[label] = v

f1 = ctk.CTkFrame(indicator_card, fg_color="transparent")
f1.pack(fill="x", padx=18, pady=(8, 4))
ctk.CTkLabel(f1, text="Insomnia Pattern", font=(FONT_FAMILY, 12),
             text_color=("#64748b", "#9aa3b2")).pack(side="left")
insomnia_chip = ctk.CTkLabel(f1, text="  —  ", font=(FONT_FAMILY, 11, "bold"),
                             fg_color=("#e5e7eb", "#2a2c38"), corner_radius=8,
                             text_color="#ffffff")
insomnia_chip.pack(side="right")

f2 = ctk.CTkFrame(indicator_card, fg_color="transparent")
f2.pack(fill="x", padx=18, pady=(4, 16))
ctk.CTkLabel(f2, text="Fragmented Sleep", font=(FONT_FAMILY, 12),
             text_color=("#64748b", "#9aa3b2")).pack(side="left")
fragmented_chip = ctk.CTkLabel(f2, text="  —  ", font=(FONT_FAMILY, 11, "bold"),
                               fg_color=("#e5e7eb", "#2a2c38"), corner_radius=8,
                               text_color="#ffffff")
fragmented_chip.pack(side="right")

# Execution log
log_card = card(right)
log_card.pack(fill="x", pady=(0, 4))
heading(log_card, "Execution Log")
log_box = ctk.CTkTextbox(log_card, font=("Consolas", 11), corner_radius=10,
                         height=260, fg_color=("#0f172a", "#0b0d14"),
                         text_color="#cbd5e1", wrap="word")
log_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
log_box.insert("end", "Ready. Select an EEG file to begin.\n")
log_box.configure(state="disabled")


# ---- Status bar --------------------------------------------------------------
footer = ctk.CTkFrame(root, corner_radius=0, height=34, fg_color=("#0f172a", "#0b0d14"))
footer.grid(row=2, column=0, sticky="ew")
footer.grid_propagate(False)
footer.grid_columnconfigure(0, weight=1)
status_label = ctk.CTkLabel(footer, text="Ready.", font=(FONT_FAMILY, 11),
                            text_color="#94a3b8", anchor="w")
status_label.grid(row=0, column=0, sticky="w", padx=18)
ctk.CTkLabel(footer, text="Preliminary indicators only — not a clinical diagnosis.",
             font=(FONT_FAMILY, 10), text_color="#64748b", anchor="e").grid(
    row=0, column=1, sticky="e", padx=18)


root.mainloop()