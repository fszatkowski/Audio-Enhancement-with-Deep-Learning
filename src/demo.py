import json
import os
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox

import librosa
import numpy as np

from common.transformations import *
from inference.audio_processor import AudioProcessor


class Demo:
    def __init__(self):
        self.MAX_STR_SIZE = 30

        self.INPUT_SR = 22050
        self.TARGET_SR = 44100
        self.DEFAULT_MODEL = "models/autoencoder/final_997_none/model.pt"

        self.audio_processor = self._init_audio_processor(self.DEFAULT_MODEL)

        self.current_orig_audio = None
        self.current_noisy_audio = None
        self.current_processed_audio = None

        self.noise_options = ["No noise", "Gaussian", "Uniform", "Impulse", "Zero mask"]

        window = tk.Tk()
        window.title("DAE")

        """ IO """
        files_frame = tk.LabelFrame(window, text="IO")
        files_frame.grid(row=0, column=0)

        """ IO DISPLAY """
        io_display_frame = tk.Frame(files_frame)
        io_display_frame.grid(row=0, column=0)

        """ INPUT FILE """
        self.input_file_label_text = tk.StringVar(value=f"Current input file:")
        input_file_label = tk.Label(
            io_display_frame, textvariable=self.input_file_label_text, justify=tk.LEFT
        )
        input_file_label.grid(row=0, column=0)

        """ MODEL FILE """
        self.model_file_label_text = tk.StringVar(
            value=f"Current model file:\n{self.DEFAULT_MODEL}"
        )
        model_file_label = tk.Label(
            io_display_frame, textvariable=self.model_file_label_text, justify=tk.LEFT
        )
        model_file_label.grid(row=1, column=0)

        """ IO BUTTONS  """
        io_buttons_frame = tk.Frame(files_frame)
        io_buttons_frame.grid(row=1, column=0)

        """ LOAD AUDIO """
        load_file_button = tk.Button(
            io_buttons_frame,
            text="Load file",
            command=lambda: self.on_load_file_button_clicked(),
        )
        load_file_button.grid(row=0, column=0)

        """ LOAD MODEL """
        load_model_button = tk.Button(
            io_buttons_frame,
            text="Load model",
            command=lambda: self.on_load_model_button_clicked(),
        )
        load_model_button.grid(row=0, column=1)

        """ SAVE AUDIO """
        save_audio_button = tk.Button(
            io_buttons_frame,
            text="Save audio",
            command=lambda: self.on_save_audio_button_clicked(),
        )
        save_audio_button.grid(row=0, column=2)

        """ NOISES """
        noise_frame = tk.LabelFrame(window, text="Noise")
        noise_frame.grid(row=0, column=1)

        """ CURRENT NOISE FRAME """
        current_noise_frame = tk.Frame(noise_frame)
        current_noise_frame.pack()

        noise_label_text = tk.StringVar(value="Current noise:")
        noise_label = tk.Label(
            current_noise_frame, textvariable=noise_label_text, justify=tk.LEFT
        )
        noise_label.grid(row=0, column=0)

        self.noise_var = tk.StringVar(value=self.noise_options[0])
        noise_options = tk.OptionMenu(
            current_noise_frame, self.noise_var, *self.noise_options
        )
        noise_options.grid(row=0, column=1)

        """ NOISE PROPS """
        noise_stats_frame = tk.Frame(noise_frame)
        noise_stats_frame.pack()

        """ NOISE MEAN INPUT """
        noise_mean_var_frame = tk.Frame(noise_stats_frame)
        noise_mean_var_frame.grid(column=0, row=0)

        noise_mean_var_label_text = tk.StringVar(value="Noise mean")
        noise_mean_var_label = tk.Label(
            noise_mean_var_frame,
            textvariable=noise_mean_var_label_text,
            justify=tk.LEFT,
        )
        noise_mean_var_label.grid(row=0, column=0)

        self.noise_mean_var = tk.DoubleVar(value=0.0)
        noise_mean_var_entry = tk.Entry(
            noise_mean_var_frame, textvariable=self.noise_mean_var
        )
        noise_mean_var_entry.grid(row=0, column=1)

        """ NOISE STD INPUT """
        noise_std_var_frame = tk.Frame(noise_stats_frame)
        noise_std_var_frame.grid(column=0, row=1)

        noise_std_var_label_text = tk.StringVar(value="Noise std")
        noise_std_var_label = tk.Label(
            noise_std_var_frame, textvariable=noise_std_var_label_text, justify=tk.LEFT
        )
        noise_std_var_label.grid(row=0, column=0)

        self.noise_std_var = tk.DoubleVar(value=0.1)
        noise_std_var_entry = tk.Entry(
            noise_std_var_frame, textvariable=self.noise_std_var
        )
        noise_std_var_entry.grid(row=0, column=1)

        """ NOISE AMPL INPUT """
        noise_amp_var_frame = tk.Frame(noise_stats_frame)
        noise_amp_var_frame.grid(column=0, row=2)

        noise_amp_var_label_text = tk.StringVar(value="Noise amplitude")
        noise_amp_var_label = tk.Label(
            noise_amp_var_frame, textvariable=noise_amp_var_label_text, justify=tk.LEFT
        )
        noise_amp_var_label.grid(row=0, column=0)

        self.noise_amp_var = tk.DoubleVar(value=0.1)
        noise_amp_var_entry = tk.Entry(
            noise_amp_var_frame, textvariable=self.noise_amp_var
        )
        noise_amp_var_entry.grid(row=0, column=1)

        """ NOISE PERC INPUT """
        noise_perc_var_frame = tk.Frame(noise_stats_frame)
        noise_perc_var_frame.grid(column=0, row=3)

        noise_perc_var_label_text = tk.StringVar(value="Noise perc")
        noise_perc_var_label = tk.Label(
            noise_perc_var_frame,
            textvariable=noise_perc_var_label_text,
            justify=tk.LEFT,
        )
        noise_perc_var_label.grid(row=0, column=0)

        self.noise_perc_var = tk.DoubleVar(value=100.0)
        noise_perc_var_entry = tk.Entry(
            noise_perc_var_frame, textvariable=self.noise_perc_var
        )
        noise_perc_var_entry.grid(row=0, column=1)

        """ NOISE BUTTONS """
        noise_buttons_frame = tk.Frame(noise_frame)
        noise_buttons_frame.pack()

        apply_noise_button = tk.Button(
            noise_buttons_frame,
            text="Apply noise",
            command=lambda: self.on_apply_noise_button_clicked(),
        )
        apply_noise_button.grid(row=0, column=0)

        """ PLAYBACK BUTTONS """
        playback_frame = tk.LabelFrame(window, text="Play")
        playback_frame.grid(row=1, column=1)

        play_orig_button = tk.Button(
            playback_frame,
            text="Original file",
            command=lambda: self.on_play_button_clicked(self.current_orig_audio),
        )
        play_orig_button.grid(row=0, column=0)

        play_noisy_button = tk.Button(
            playback_frame,
            text="Noisy file",
            command=lambda: self.on_play_button_clicked(self.current_noisy_audio),
        )
        play_noisy_button.grid(row=0, column=1)

        play_proc_button = tk.Button(
            playback_frame,
            text="Processed file",
            command=lambda: self.on_play_button_clicked(self.current_processed_audio),
        )
        play_proc_button.grid(row=0, column=2)

        """ PROCESS BUTTON """
        proc_save_frame = tk.LabelFrame(window, text="Inference")
        proc_save_frame.grid(row=1, column=0)

        process_button = tk.Button(
            proc_save_frame,
            text="Process audio",
            command=lambda: self.on_process_button_clicked(),
        )
        process_button.pack()

        window.mainloop()

    def on_load_file_button_clicked(self):
        input_file = tk.filedialog.askopenfilename()
        try:
            self.current_orig_audio, _ = librosa.load(
                input_file, sr=self.INPUT_SR, mono=False
            )
            self.input_file_label_text.set(
                f"Current input file:\n{self._crop_filename(input_file)}"
            )
        except Exception as e:
            tk.messagebox.showinfo("Error", "Incorrect audio file provided")

    def on_load_model_button_clicked(self):
        model_file = tk.filedialog.askopenfilename()
        self.model_file_label_text.set(
            f"Current model file:\n{self._crop_filename(model_file)}"
        )
        self.audio_processor = self._init_audio_processor(model_file)

    @staticmethod
    def _init_audio_processor(model_file: str) -> AudioProcessor:
        metadata_file = os.path.join(os.path.dirname(model_file), "metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return AudioProcessor(metadata, batch_size=1)

    def on_save_audio_button_clicked(self):
        save_file = tk.filedialog.askopenfilename()
        if self.current_processed_audio is None:
            tk.messagebox.showinfo(
                "Error",
                "No output file found\nPlease create output before using this button",
            )
        else:
            librosa.output.write_wav(
                save_file,
                y=np.asfortranarray(np.squeeze(self.current_processed_audio)),
                sr=self.TARGET_SR,
            )

    def _crop_filename(self, filename: str) -> str:
        if len(filename) > self.MAX_STR_SIZE:
            return "..." + filename[-self.MAX_STR_SIZE :]
        return filename

    def on_process_button_clicked(self):
        if self.current_noisy_audio is None:
            tk.messagebox.showinfo(
                "Error",
                "No noisy audio provided\nLoad audio and apply noise before processing",
            )
            return
        self.current_processed_audio = self.audio_processor.process_array(
            self.current_noisy_audio
        )

    def on_apply_noise_button_clicked(self):
        mean = self.noise_mean_var.get()
        std = self.noise_std_var.get()
        amplitude = self.noise_amp_var.get()
        noise_perc = self.noise_perc_var.get()

        if self.noise_var == "No noise":
            self.current_noisy_audio = self.current_orig_audio
        elif self.noise_var == "Gaussian":
            self.current_noisy_audio = GaussianNoisePartial(
                apply_probability=1, mean=mean, std=std, noise_percent=noise_perc
            )
        elif self.noise_var == "Uniform":
            self.current_noisy_audio = UniformNoisePartial(
                apply_probability=1, amplitude=amplitude, noise_percent=noise_perc
            )
        elif self.noise_var == "Impulse":
            tk.messagebox.showinfo("Error", "Not implemented yet")
        elif self.noise_var == "Zero mask":
            self.current_noisy_audio = ZeroSamplesTransformation(
                apply_probability=1, noise_percent=noise_perc
            )

    @staticmethod
    def _validate_value(
        value: float, min_val: float, max_val: float, name: str
    ) -> bool:
        if min_val <= value <= max_val:
            return True
        else:
            tk.messagebox.showinfo(
                "Incorrect value",
                f"Value of {name}: {value} incorrect\nValue must be between {min_val} and {max_val}",
            )
            return False

    def on_play_button_clicked(self, audio: np.array):
        # TODO do it
        pass


if __name__ == "__main__":
    Demo()
