import tkinter as tk
import tkinter.filedialog
import numpy as np


class GUI:
    def __init__(self):
        self.MAX_STR_SIZE = 30
        self.current_orig_audio = None
        self.current_processed_audio = None

        self.model = None

        window = tk.Tk()
        window.columnconfigure([0, 1], minsize=250)
        window.rowconfigure([0, 1], minsize=50)

        self.input_file_label_text = tk.StringVar()
        self.input_file_label_text.set(f"Input file:")
        self.input_file_label = tk.Label(textvariable=self.input_file_label_text, justify=tk.LEFT)
        self.input_file_label.grid(row=0, column=0)

        self.model_file_label_text = tk.StringVar()
        self.model_file_label_text.set(f"Model file:")
        self.model_file_label = tk.Label(textvariable=self.model_file_label_text, justify=tk.LEFT)
        self.model_file_label.grid(row=0, column=1)

        """
        TODO:
            -add noise and noise selection
            -implement model and audio loading
            -implement model and audio playback
            -add process button for running inference
            -add 'add noise' button
            -add button for saving results
        """
        # noise_label = tk.Label(text="Noise").pack()
        #
        load_file_button = tk.Button(text="Load file", command=lambda: self.on_load_file_button_clicked())
        load_file_button.grid(row=1, column=0)

        load_model_button = tk.Button(text="Load model", command=lambda: self.on_load_model_button_clicked())
        load_model_button.grid(row=1, column=1)

        # process_button = tk.Button(text="Process").pack()
        # save_button = tk.Button(text="Save").pack()
        # play_orig_button = tk.Button(text="Play").pack()
        # play_result_button = tk.Button(text="Play").pack()

        window.mainloop()

    def on_load_file_button_clicked(self):
        input_file = tk.filedialog.askopenfilename()
        self.input_file_label_text.set(f"Input file:\n{self.crop_filename(input_file)}")
        self.current_orig_audio = load_audio_from_file(input_file)

    def on_load_model_button_clicked(self):
        model_file = tk.filedialog.askopenfilename()

        self.model_file_label_text.set(f"Model file:\n{self.crop_filename(model_file)}")
        self.model = load_model_from_file(model_file)

    def crop_filename(self, filename: str)->str:
        if len(filename) > self.MAX_STR_SIZE:
            return "..." + filename[-self.MAX_STR_SIZE:]
        return filename

# Currently mock methods
def load_audio_from_file(path: str) -> np.array:
    return np.zeros((10))

def load_model_from_file(model_file_path: str) -> "torch.nn.Module":
    return None



def main():
    gui = GUI()


if __name__ == '__main__':
    main()
