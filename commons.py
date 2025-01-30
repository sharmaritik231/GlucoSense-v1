import constants
import numpy as np
import pandas as pd
import tkinter as tk
import feature_eng
from tkinter import messagebox
import pickle
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.base import TransformerMixin, BaseEstimator

def load_model(file_path):
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)

    return model


def save_labels_as_pdf(widgets, name):
    # Create a new PDF file
    pdf_filename = f"{name}.pdf"

    # Setup the canvas for the PDF with A4 page size
    c = canvas.Canvas(pdf_filename, pagesize=A4)

    # Define the starting position for drawing the content
    x = 100  # X-coordinate
    y = 800  # Start from the top of the page

    # Loop through the widgets and write their content into the PDF
    for widget in widgets:
        if isinstance(widget, tk.Label):  # If it's a Label, get its text
            text = widget.cget("text")
            c.drawString(x, y, str(text))  # Ensure text is a string
            y -= 20  # Move down for the next line
        elif isinstance(widget, tk.Frame):  # If it's a Frame, extract labels from it
            for child in widget.winfo_children():
                if isinstance(child, tk.Label):
                    text = child.cget("text")
                    c.drawString(x, y, str(text))  # Ensure text is a string
                    y -= 20

    # Save and close the PDF
    c.showPage()
    c.save()

    # Show a message box confirming the PDF save
    messagebox.showinfo("Save as PDF", f"PDF saved successfully as '{pdf_filename}'")


# Function to display the patient decision in a tkinter window
def project_decision_in_app(decision1, name, body_vitals, bgl=None):
    # Create the main window
    window = tk.Tk()
    window.title("Diagnostic Report")

    # Modern color palette inspired by Coolors
    bg_color = "#F8F1F1"  # Light background color
    header_color = "#6A0572"  # Deep purple for headers
    accent_color = "#669bbc"  # Bright pink for buttons
    text_color = "#3E2C41"  # Dark purple for text
    button_hover_color = "#FF82A9"  # Lighter shade for button hover

    # Font styles
    font_title = ("Arial", 22, "bold")
    font_subtitle = ("Arial", 18, "bold")
    font_label = ("Arial", 14, "bold")
    font_value = ("Arial", 14)

    window.configure(bg=bg_color)
    gender = 'Male' if body_vitals['Gender'][0] == 0 else 'Female'

    # Create a label widget for the title
    title_label = tk.Label(window, text="Glucometer Prediction Result", font=font_title, fg=header_color, bg=bg_color)
    title_label.pack(pady=20)

    # Display patient details
    patient_frame = tk.Frame(window, bg=bg_color)
    patient_frame.pack(anchor='w', padx=10, pady=6)

    def add_label_value(frame, label_text, value_text, row):
        tk.Label(frame, text=label_text, font=font_label, fg=text_color, bg=bg_color).grid(row=row, column=0, sticky='w')
        tk.Label(frame, text=str(value_text), font=font_value, fg=text_color, bg=bg_color).grid(row=row, column=1, sticky='w')

    tk.Label(patient_frame, text="Patient Details:", font=font_subtitle, fg=header_color, bg=bg_color).grid(row=0, column=0, columnspan=2, sticky='w', pady=6)
    add_label_value(patient_frame, "Name:", name, 1)
    add_label_value(patient_frame, "Age:", body_vitals['Age'][0], 2)
    add_label_value(patient_frame, "Gender:", gender, 3)
    add_label_value(patient_frame, "Heart Rate:", body_vitals['Heart_Beat'][0], 4)
    add_label_value(patient_frame, "SPO2:", body_vitals['SPO2'][0], 5)
    add_label_value(patient_frame, "Blood Pressure:", f"{body_vitals['max_BP'][0]} | {body_vitals['min_BP'][0]}", 6)

    # Display final results
    result_frame = tk.Frame(window, bg=bg_color)
    result_frame.pack(anchor='w', padx=10, pady=6)

    tk.Label(result_frame, text="Final Results:", font=font_subtitle, fg=header_color, bg=bg_color).grid(row=0, column=0, columnspan=2, sticky='w', pady=6)
    tk.Label(result_frame, text=f"{name}, {decision1}", font=font_value, fg=text_color, bg=bg_color).grid(row=1, column=0, columnspan=2, sticky='w')

    if bgl is not None:
        tk.Label(result_frame, text="Blood Glucose Level (mg/dL):", font=font_label, fg=text_color, bg=bg_color).grid(row=2, column=0, sticky='w')
        tk.Label(result_frame, text=str(bgl), font=font_value, fg=text_color, bg=bg_color).grid(row=2, column=1, sticky='w')

    # Save button
    def on_button_enter(e):
        save_button['bg'] = button_hover_color

    def on_button_leave(e):
        save_button['bg'] = accent_color

    save_button = tk.Button(window, text="Save as PDF", font=font_value, fg="white", bg=accent_color, activebackground=button_hover_color, command=lambda: messagebox.showinfo("Save", "PDF saved successfully!"))
    save_button.pack(pady=20)
    save_button.bind("<Enter>", on_button_enter)
    save_button.bind("<Leave>", on_button_leave)

    # Set the geometry
    app_width = 550
    app_height = 650
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width / 2) - (app_width / 2)
    y = (screen_height / 2) - (app_height / 2)
    window.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')

    # Start the main event loop
    window.mainloop()


def perform_feature_selection(test_data):
    selector = load_model("models/selector.pkl")
    names = test_data.columns[selector.get_support()]
    return pd.DataFrame(data=selector.transform(test_data), columns=names)


def perform_diabetes_test(test_data):
    model = load_model('models/stack.pkl')
    test_label = model.predict(test_data)

    if test_label == 0:
        final_message = "Your Blood Sugar is Low"
    elif test_label == 1:
        final_message = "Your Blood Sugar is Medium"
    else:
        final_message = "Your Blood sugar is High"

    return final_message


def perform_bgl_test(test_data):
    scaler = load_model("models/coll_scaler.pkl")
    features = scaler.transform(test_data)

    bgl_model = load_model(file_path='models/AdaBoost.pkl')
    bgl_value = bgl_model.predict(features)[0]
    return np.round(bgl_value, 2)


def get_body_vitals_info():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender - Press 1 for Female and 0 for Male: "))
    heart_rate = int(input("Enter heart rate: "))
    max_bp = int(input("Enter Max BP: "))
    min_bp = int(input("Enter Min BP: "))
    spo2 = int(input("Enter SPO2 value: "))
    data = {'Age': [age], 'Gender': [gender], 'Heart_Beat': [heart_rate], 'SPO2': [spo2], 'max_BP': [max_bp], 'min_BP': [min_bp]}
    return pd.DataFrame(data)


def remove_irrelevant_data(file_path):
    # Read the CSV file into a DataFrame, skipping the first 3 rows and setting the 4th row as header
    df = pd.read_csv(file_path, skiprows=3)
    df = df.iloc[:, 1:]     # Remove the first column

    # Set the correct column headers
    df.columns = ['H', 'MQ138', 'MQ2', 'SSID', 'T', 'TGS2600', 'TGS2602', 'TGS2603', 'TGS2610', 'TGS2611', 'TGS2620', 'TGS822', 'Device', 'Time']
    df = df.drop(['SSID', 'Device', 'H', 'T', 'Time'], axis=1)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_data():
    cleaned_df = remove_irrelevant_data(constants.TEST_DATA_DIR)
    body_vitals = get_body_vitals_info()
    features_df = feature_eng.generate_features(df=cleaned_df)
    final_df = pd.concat([body_vitals, features_df], axis=1)
    return final_df, body_vitals


class RemoveHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Calculate the correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the specified threshold
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X, y=None):
        # Drop the highly correlated features
        if self.to_drop_ is not None:
            return X.drop(columns=self.to_drop_)
        else:
            return X

    def fit_transform(self, X, y=None, **fit_params):
        # Use the fit method and then the transform method
        return super().fit_transform(X, y, **fit_params)

    def __getstate__(self):
        # Return the object's state as a dictionary
        return self.__dict__

    def __setstate__(self, state):
        # Restore the object's state from the dictionary
        self.__dict__.update(state)