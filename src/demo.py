import tkinter as tk
import tkinter.filedialog


class Demo:
    def __init__(self):
        self.MAX_STR_SIZE = 30

        self.model = None

        self.current_orig_audio = None
        self.current_noisy_audio = None
        self.current_processed_audio = None

        self.noise_options = ["No noise", "Gaussian", "Uniform", "Impulse", "Zero mask"]

        window = tk.Tk()
        window.title("DAE")

        """ FILE OPERATIONS """
        files_frame = tk.LabelFrame(window, text="IO")
        files_frame.grid(row=0, column=0)

        """ INPUT FILE """
        input_file_frame = tk.Frame(files_frame)
        input_file_frame.grid(row=0, column=0)

        self.input_file_label_text = tk.StringVar(value=f"Input file:")
        input_file_label = tk.Label(input_file_frame, textvariable=self.input_file_label_text, justify=tk.LEFT)
        input_file_label.grid(row=0, column=0)

        load_file_button = tk.Button(input_file_frame, text="Load file",
                                     command=lambda: self.on_load_file_button_clicked())
        load_file_button.grid(row=1, column=0)

        """ MODEL FILE """
        model_file_frame = tk.Frame(files_frame)
        model_file_frame.grid(row=1, column=0)

        self.model_file_label_text = tk.StringVar(value=f"Model file:")
        model_file_label = tk.Label(model_file_frame, textvariable=self.model_file_label_text, justify=tk.LEFT)
        model_file_label.grid(row=0, column=1)

        load_model_button = tk.Button(model_file_frame, text="Load model",
                                      command=lambda: self.on_load_model_button_clicked())
        load_model_button.grid(row=1, column=1)

        """ SAVE AUDIO FILE """
        save_audio_frame = tk.Frame(files_frame)
        save_audio_frame.grid(row=2, column=0)

        self.save_audio_label_text = tk.StringVar(value=f"Save audio to file:")
        save_audio_label = tk.Label(save_audio_frame, textvariable=self.save_audio_label_text, justify=tk.LEFT)
        save_audio_label.grid(row=0, column=1)

        save_audio_button = tk.Button(save_audio_frame, text="Save audio",
                                      command=lambda: self.on_save_audio_button_clicked())
        save_audio_button.grid(row=1, column=1)

        """ NOISES """
        noise_frame = tk.LabelFrame(window, text="Noise")
        noise_frame.grid(row=0, column=1)

        """ CURRENT NOISE FRAME """
        current_noise_frame = tk.Frame(noise_frame)
        current_noise_frame.pack()

        noise_label_text = tk.StringVar(value="Current noise:")
        noise_label = tk.Label(current_noise_frame, textvariable=noise_label_text, justify=tk.LEFT)
        noise_label.grid(row=0, column=0)

        init_noise_var = tk.StringVar(value=self.noise_options[0])
        noise_options = tk.OptionMenu(current_noise_frame, init_noise_var, *self.noise_options)
        noise_options.grid(row=0, column=1)

        noise_stats_frame = tk.Frame(noise_frame)
        noise_stats_frame.pack()

        """ NOISE MEAN INPUT """
        noise_mean_var_frame = tk.Frame(noise_stats_frame)
        noise_mean_var_frame.grid(column=0, row=0)

        noise_mean_var_label_text = tk.StringVar(value="Noise mean")
        noise_mean_var_label = tk.Label(noise_mean_var_frame, textvariable=noise_mean_var_label_text, justify=tk.LEFT)
        noise_mean_var_label.grid(row=0, column=0)

        self.noise_mean_var = tk.DoubleVar(value=0.0)
        noise_mean_var_entry = tk.Entry(noise_mean_var_frame, textvariable=self.noise_mean_var)
        noise_mean_var_entry.grid(row=0, column=1)

        """ NOISE STD INPUT """
        noise_std_var_frame = tk.Frame(noise_stats_frame)
        noise_std_var_frame.grid(column=0, row=1)

        noise_std_var_label_text = tk.StringVar(value="Noise std")
        noise_std_var_label = tk.Label(noise_std_var_frame, textvariable=noise_std_var_label_text, justify=tk.LEFT)
        noise_std_var_label.grid(row=0, column=0)

        self.noise_std_var = tk.DoubleVar(value=1.0)
        noise_std_var_entry = tk.Entry(noise_std_var_frame, textvariable=self.noise_std_var)
        noise_std_var_entry.grid(row=0, column=1)

        """ NOISE PERC INPUT """
        noise_perc_var_frame = tk.Frame(noise_stats_frame)
        noise_perc_var_frame.grid(column=0, row=2)

        noise_perc_var_label_text = tk.StringVar(value="Noise perc")
        noise_perc_var_label = tk.Label(noise_perc_var_frame, textvariable=noise_perc_var_label_text, justify=tk.LEFT)
        noise_perc_var_label.grid(row=0, column=0)

        self.noise_perc_var = tk.DoubleVar(value=100.0)
        noise_perc_var_entry = tk.Entry(noise_perc_var_frame, textvariable=self.noise_perc_var)
        noise_perc_var_entry.grid(row=0, column=1)

        """ NOISE BUTTONS """
        noise_buttons_frame = tk.Frame(noise_frame)
        noise_buttons_frame.pack()

        apply_noise_button = tk.Button(noise_buttons_frame, text="Apply noise",
                                       command=lambda: self.on_apply_noise_button_clicked())
        apply_noise_button.grid(row=0, column=0)

        clear_noise_button = tk.Button(noise_buttons_frame, text="Clear noise",
                                       command=lambda: self.on_clear_noise_button_clicked())
        clear_noise_button.grid(row=0, column=1)

        """ PLAYBACK BUTTONS """
        playback_frame = tk.LabelFrame(window, text="Play")
        playback_frame.grid(row=1, column=1)

        play_orig_button = tk.Button(playback_frame, text="Original file",
                                     command=lambda: self.on_play_orig_button_clicked())
        play_orig_button.grid(row=0, column=0)

        play_noisy_button = tk.Button(playback_frame, text="Noisy file",
                                      command=lambda: self.on_play_noisy_button_clicked())
        play_noisy_button.grid(row=0, column=1)

        play_proc_button = tk.Button(playback_frame, text="Processed file",
                                     command=lambda: self.on_play_proc_button_clicked())
        play_proc_button.grid(row=0, column=2)

        """ PROCESS BUTTON """
        proc_save_frame = tk.LabelFrame(window, text="Inference")
        proc_save_frame.grid(row=1, column=0)

        process_button = tk.Button(proc_save_frame, text="Process audio",
                                   command=lambda: self.on_process_button_clicked())
        process_button.pack()

        window.mainloop()

    def on_load_file_button_clicked(self):
        input_file = tk.filedialog.askopenfilename()

        self.input_file_label_text.set(f"Input file: {self._crop_filename(input_file)}")
        # TODO load audio

    def on_load_model_button_clicked(self):
        model_file = tk.filedialog.askopenfilename()

        self.model_file_label_text.set(f"Model file: {self._crop_filename(model_file)}")
        # TODO load model

    def on_save_audio_button_clicked(self):
        save_file = tk.filedialog.askopenfilename()

        self.save_audio_label_text.set(f"Save audio to file: {self._crop_filename(save_file)}")
        # TODO save processed audio to file

    def _crop_filename(self, filename: str) -> str:
        if len(filename) > self.MAX_STR_SIZE:
            return "..." + filename[-self.MAX_STR_SIZE:]
        return filename


if __name__ == '__main__':
    Demo()
