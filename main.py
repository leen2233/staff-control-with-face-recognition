import random
import re
import shutil
import tkinter
from datetime import datetime, time, timedelta

import PIL
import customtkinter
from tkinter import filedialog

import cv2
import face_recognition
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from tkinter import messagebox
from tkcalendar import DateEntry
from cv2 import VideoCapture, imwrite
import sqlite3

if not os.path.exists("known_people"):
    os.makedirs("known_people")

if not os.path.exists("records"):
    os.makedirs("records")

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                  (id INTEGER PRIMARY KEY, person TEXT, date TEXT, time TEXT, image_path TEXT)''')

# Commit changes and close connection
conn.commit()
conn.close()
vid = VideoCapture(0)
width, height = 800, 600

# Set the width and height
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            print(
                "WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            print("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(
            known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:
            return known_names[result.index(True)]
        else:
            return "unknown_people"

    if not unknown_encodings:
        # print out fact that no faces were found in image
        return False


def test_image_encoding(image_to_check):
    unknown_image = face_recognition.load_image_file(image_to_check)
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)
    return unknown_encodings


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


# main("known_people", "obama2.jpg")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.known_face_data = [None, None]
        self.process_this_frame = None
        self.home_frame_label = None
        self.image_2 = None
        self.shot_picture_2 = None
        self.selected_image_file_path = None
        self.selected_image = None
        self.title("Kärhananyň işgärler hasabaty")
        self.geometry("700x450")
        self.row_counter = 2
        self.entries = []
        self.radio_var = tkinter.IntVar(value=1)
        self.image_1 = None
        self.shot_picture = None
        self.camera = 0
        self.already_added_list = []

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "images")

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame,
                                                             text="Kärhananyň \nişgärler hasabaty",
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10,
                                                   text="Surata al",
                                                   fg_color="transparent", text_color=("gray10", "gray90"),
                                                   hover_color=(
                                                       "gray70", "gray30"),
                                                   anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="Işgärleri Gör",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=(
                                                          "gray70", "gray30"),
                                                      anchor="w", command=self.frame_2_button_event)
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.add_staff_frame_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                              border_spacing=10, text="Işgär Goş",
                                                              fg_color="transparent", text_color=("gray10", "gray90"),
                                                              hover_color=(
                                                                  "gray70", "gray30"),
                                                              anchor="w", command=self.frame_3_button_event)
        self.add_staff_frame_button.grid(row=3, column=0, sticky="ew")

        self.list_detections_frame_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                                    border_spacing=10, text="Işgärler",
                                                                    fg_color="transparent",
                                                                    text_color=(
                                                                        "gray10", "gray90"),
                                                                    hover_color=(
                                                                        "gray70", "gray30"),
                                                                    anchor="w",
                                                                    command=self.list_detection_button_event)
        self.list_detections_frame_button.grid(row=4, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame,
                                                                values=[
                                                                    "Dark", "Light", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(
            row=6, column=0, padx=20, pady=20, sticky="s")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_large_image_label = customtkinter.CTkLabel(
            self.home_frame, text="Baş sahypa")
        self.home_frame_large_image_label.grid(
            row=1, column=0, padx=20, pady=10)

        self.home_frame_take_picture = customtkinter.CTkButton(self.home_frame, text="Kamerany aç",
                                                               command=self.open_camera_2_callback)
        self.home_frame_take_picture.grid(row=2, column=0, padx=20, pady=10)
        self.home_frame_take_picture = customtkinter.CTkButton(self.home_frame, text="Kamerany duruz",
                                                               command=self.close_camera_2_callback)
        self.home_frame_take_picture.grid(row=3, column=0, padx=20, pady=10)

        # create second frame
        self.second_frame = customtkinter.CTkScrollableFrame(
            self, corner_radius=0, fg_color="transparent")
        self.second_frame.grid_columnconfigure(0, weight=2)
        self.second_frame_label = customtkinter.CTkLabel(
            self.second_frame, text="Işgärler")
        self.second_frame_label.grid(row=0, column=0, padx=20, pady=10)

        image_files = [f for f in os.listdir("known_people") if f.endswith(
            ('.jpg', '.jpeg', '.png', '.gif'))]
        for file in image_files:
            image_path = os.path.join("known_people", file)
            self.second_frame_selected_image = customtkinter.CTkImage(
                Image.open(image_path), size=(200, 200))

            self.second_frame_label = customtkinter.CTkLabel(self.second_frame, text="    " + os.path.splitext(file)[0],
                                                             image=self.second_frame_selected_image,
                                                             compound="left")
            self.second_frame_label.grid(
                row=self.row_counter, column=0, padx=20, pady=20)
            delete_button = customtkinter.CTkButton(
                self.second_frame, text="Poz", fg_color="red")
            delete_button.grid(row=self.row_counter,
                               column=1, padx=20, pady=20)
            delete_button.configure(
                command=lambda label=(file, self.second_frame_label), to_delete_button=delete_button: self.delete_image(
                    label, to_delete_button))
            self.row_counter += 1

        # create third frame
        self.third_frame = customtkinter.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.third_frame.grid_columnconfigure(0, weight=1)
        self.list_detections_frame = customtkinter.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.list_detections_frame.grid_columnconfigure(0, weight=1)

        self.start_label = customtkinter.CTkLabel(
            self.list_detections_frame, text="Senäni saýlaň:")
        self.start_label.grid(row=0, column=0, padx=5, pady=5)

        self.start_cal = DateEntry(self.list_detections_frame)
        self.start_cal.grid(row=1, column=0, padx=5, pady=5)

        self.view_button = customtkinter.CTkButton(self.list_detections_frame, text="Ýazgylary gör",
                                                   command=self.view_entries)
        self.view_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.entries_frame = customtkinter.CTkScrollableFrame(self.list_detections_frame, corner_radius=0,
                                                              fg_color="transparent", width=700)
        self.entries_frame.grid(row=6, column=0, columnspan=5, padx=5, pady=5)

        self.third_frame_label = customtkinter.CTkLabel(
            self.third_frame, text="Işgär Goş")
        self.third_frame_label.grid(row=1, column=0, padx=20, pady=10)
        self.third_frame_select_image = customtkinter.CTkButton(self.third_frame, text="Kamerany aç",
                                                                command=self.open_camera_1_callback)
        self.third_frame_select_image.grid(row=2, column=0, padx=20, pady=10)
        self.third_frame_select_image = customtkinter.CTkButton(self.third_frame, text="Kamerany duruz",
                                                                command=self.close_camera_1_callback)
        self.third_frame_select_image.grid(row=3, column=0, padx=20, pady=10)

        self.third_frame_take_picture = customtkinter.CTkButton(self.third_frame, text="Surata al",
                                                                command=self.third_frame_shot_picture)
        self.third_frame_take_picture.grid(row=4, column=0, padx=20, pady=10)

        self.third_frame_staff_name = customtkinter.CTkEntry(
            self.third_frame, placeholder_text="Ady Familiyasy")
        self.third_frame_staff_name.grid(row=5, column=0, padx=20, pady=10)

        self.third_frame_staff_role = customtkinter.CTkEntry(
            self.third_frame, placeholder_text="Wezipesi")
        self.third_frame_staff_role.grid(row=6, column=0, padx=20, pady=10)

        self.third_frame_staff_time = customtkinter.CTkEntry(
                    self.third_frame, placeholder_text="Wagty")
        self.third_frame_staff_time.grid(row=7, column=0, padx=20, pady=10)

        self.third_frame_add_staff_button = customtkinter.CTkButton(self.third_frame, text="Işgär Goş",
                                                                    command=self.add_staff_to_known_people)
        self.third_frame_add_staff_button.grid(
            row=8, column=0, padx=20, pady=10)

        # select default frame
        self.select_frame_by_name("home")

    def view_entries(self):
        start_date = self.start_cal.get_date()
        entries = self.get_entries(start_date)
        for entry in self.entries:
            entry.destroy()
        row_counter = 1
        for entry in entries:
            id, name, date, _time, image_path, color = entry
            print(color)
            entries_image = customtkinter.CTkImage(
                Image.open(image_path), size=(200, 200))

            entries_label = customtkinter.CTkLabel(self.entries_frame, text="    " + name,
                                                   image=entries_image, text_color=(
                                                       color, color),
                                                   compound="left")
            entries_label.grid(row=row_counter, column=0, padx=20, pady=20)

            entries_label_2 = customtkinter.CTkLabel(
                self.entries_frame, text=date + " " + _time)
            entries_label_2.grid(row=row_counter, column=1, padx=20, pady=20)
            self.entries.append(entries_label)
            self.entries.append(entries_label_2)
            row_counter += 1

    def get_entries(self, date):
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()

        # Execute SQL query to retrieve records within the given time range
        cursor.execute('SELECT * FROM detections WHERE date = ?',
                       (date,))

        # Fetch the filtered records
        records = cursor.fetchall()

        custom = []

        for record in records:
            record_time = datetime.strptime(record[3], '%H:%M:%S').time()
            scheduled_time_str = record[1].split(' - ')[2]  # Gets "08:00"
            scheduled_time = datetime.strptime(scheduled_time_str, '%H:%M').time()
            
            record = list(record)
            # If actual time is more than 15 minutes later than scheduled time
            if record_time > (datetime.combine(datetime.today(), scheduled_time) + timedelta(minutes=15)).time():
                record.append("red")
            elif time(17, 0, 0) <= record_time <= time(19, 0, 0):
                record.append("yellow")
            else:
                record.append("green")
            custom.append(record)

        conn.close()
        return custom

    def take_picture(self):
        cam_port = 0
        cam = VideoCapture(cam_port)

        result, image = cam.read()

        if result:
            imwrite("unknown.jpg", image)
            known_names, known_face_encodings = scan_known_people(
                "known_people")
            detected_person = test_image(
                "unknown.jpg", known_names, known_face_encodings)

            if detected_person and detected_person != "unknown_people":
                self.save_to_database(detected_person)
                messagebox.showinfo(
                    "Success", f"{detected_person} ýüzi tanaldy we ýazga alyndy!")
            else:
                messagebox.showerror("Error", "Hiç bir ýüz tapylmady")
        else:
            messagebox.showerror("Error", "Kamera tapylmady!")

    def third_frame_shot_picture(self):
        self.shot_picture = True

    def open_camera_2_callback(self):
        self.camera = 2
        self.known_face_data = scan_known_people("known_people")
        self.open_camera_2()
        self.process_this_frame = True

    def close_camera_2_callback(self):
        self.camera = 0

    def open_camera_2(self):
        if self.camera == 2:
            # Capture the video frame by frame
            # self.third_frame_select_image.configure(command=self.third_frame_shot_picture, text="Surata Al")
            _, frame = vid.read()
            if _:
                if not self.shot_picture_2:
                    cv2.imwrite("temp.jpg", frame)
                    unknown_image = face_recognition.load_image_file(
                        "temp.jpg")

                    # Find all the faces and face encodings in the unknown image
                    face_locations = face_recognition.face_locations(
                        unknown_image)
                    face_encodings = face_recognition.face_encodings(
                        unknown_image, face_locations)

                    pil_image = Image.fromarray(unknown_image)
                    # Create a Pillow ImageDraw Draw instance to draw with
                    draw = ImageDraw.Draw(pil_image)

                    # Loop through each face found in the unknown image
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(
                            self.known_face_data[1], face_encoding)

                        name = "Unknown"

                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_data[0][first_match_index]
                            found = False
                            five_minute_ago = timedelta(minutes=5)
                            for i in self.already_added_list:
                                if i[0] == name:
                                    if datetime.now() - i[1] < five_minute_ago:
                                        found = True
                                        break
                                    else:
                                        self.already_added_list.remove(i)
                            if not found:
                                filename = "records/" + \
                                    str(random.randint(10000000000000,
                                        999999999999999)) + ".jpg"
                                shutil.move("temp.jpg", filename)
                                self.save_to_database(name, filename)
                                self.already_added_list.append(
                                    [name, datetime.now()])

                        # Draw a box around the face using the Pillow module
                        draw.rectangle(
                            ((left, top), (right, bottom)), outline=(0, 0, 255))

                        # Draw a label with a name below the face
                        text_height = draw.textlength(name)
                        text_height = 60
                        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255),
                                       outline=(0, 0, 255))

                        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255),
                                  font=ImageFont.truetype("utils/college.ttf", 60))

                    # Remove the drawing library from memory as per the Pillow docs
                    del draw
                    # Capture the latest frame and transform to image
                    self.image_2 = pil_image

                    self.selected_image = customtkinter.CTkImage(
                        self.image_2, size=(200, 200))
                    self.home_frame_label = customtkinter.CTkLabel(self.home_frame, text="",
                                                                   image=self.selected_image,
                                                                   compound="left")
                    self.home_frame_label.grid(
                        row=0, column=0, padx=20, pady=20)

                    # Repeat the same process after every 10 seconds
                    self.home_frame_label.after(10, self.open_camera_2)
            else:
                messagebox.showerror("Error", "Kamera tapylmady!")

    def open_camera_1_callback(self):
        self.camera = 1
        self.open_camera()

    def close_camera_1_callback(self):
        self.camera = 0

    def open_camera(self):
        if self.camera == 1:
            # Capture the video frame by frame
            _, frame = vid.read()
            if _:
                # Convert image from one color space to other
                opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                if not self.shot_picture:
                    # Capture the latest frame and transform to image
                    self.image_1 = Image.fromarray(opencv_image)

                    self.selected_image = customtkinter.CTkImage(
                        self.image_1, size=(200, 200))
                    self.navigation_frame_label = customtkinter.CTkLabel(self.third_frame, text="",
                                                                         image=self.selected_image,
                                                                         compound="left")
                    self.navigation_frame_label.grid(
                        row=0, column=0, padx=20, pady=20)

                    # Repeat the same process after every 10 seconds
                    self.navigation_frame_label.after(10, self.open_camera)
            else:
                messagebox.showerror("Error", "Kamera tapylmady!")

    def save_to_database(self, name, filename):
        # Connect to the database
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                          (id INTEGER PRIMARY KEY, person TEXT, date TEXT, time TEXT, image_path TEXT)''')

        # Insert the detected person and current time into the database
        time = datetime.now().strftime('%H:%M:%S')
        date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(
            "INSERT INTO detections (person, date, time, image_path) VALUES (?, ?, ?, ?)", (name, date, time, filename))

        # Commit changes and close connection
        conn.commit()
        conn.close()

    def delete_image(self, arr, delete_button):
        try:
            filename, label = arr
            image_path = os.path.join("known_people", filename)
            os.remove(image_path)
            label.destroy()
            delete_button.destroy()
            messagebox.showinfo("Üstünlikli", "Işgär pozuldy")
        except Exception as e:
            messagebox.showerror(
                "Näsazlyk", "Bir näsazlyk döredi. Ýazgysy: " + str(e))

    def add_staff_to_known_people(self):
        try:
            self.image_1 = self.image_1.convert("RGB")
            self.image_1.save("temp.jpg")
            is_valid_image = test_image_encoding("temp.jpg")
            if is_valid_image:
                new_name = self.third_frame_staff_name.get() + " - " + \
                    self.third_frame_staff_role.get()
                arrival_time_str = self.third_frame_staff_time.get()
                if not new_name or not arrival_time_str:
                    messagebox.showerror("Näsazlyk", "Dogry at giriziň.")
                    return
                try:
                    arrival_time = datetime.strptime(
                        arrival_time_str, '%H:%M').time()
                except ValueError:
                    messagebox.showerror(
                        "Näsazlyk", "Wagty dogry formatda giriziň (HH:MM).")
                    return
                new_name = new_name + " - " + arrival_time_str
                destination_path = os.path.join(
                                    "known_people", new_name + ".jpg")

                try:
                    self.image_1 = self.image_1.convert("RGB")
                    self.image_1.save(destination_path)
                    self.third_frame_staff_name.delete(0, "end")
                    self.third_frame_staff_role.delete(0, "end")
                    self.navigation_frame_label.destroy()
                    self.second_frame_selected_image = customtkinter.CTkImage(Image.open(destination_path),
                                                                              size=(200, 200))
                    self.second_frame_label = customtkinter.CTkLabel(self.second_frame,
                                                                     text="    " + os.path.splitext(new_name)[
                                                                         0],
                                                                     image=self.second_frame_selected_image,
                                                                     compound="left")
                    self.second_frame_label.grid(
                        row=self.row_counter, column=0, padx=20, pady=20)
                    delete_button = customtkinter.CTkButton(
                        self.second_frame, text="Delete", fg_color="red")
                    delete_button.grid(row=self.row_counter,
                                       column=1, padx=20, pady=20)
                    delete_button.configure(command=lambda label=(new_name + ".jpg", self.second_frame_label),
                                            to_delete_button=delete_button: self.delete_image(label,
                                                                                              to_delete_button))
                    self.row_counter += 1
                    messagebox.showinfo(
                        "Üstünlikli", "Işgär üstünlikli goşuldy")
                except Exception as e:
                    messagebox.showerror(
                        "Näsazlyk", "Bir näsazlyk döredi. Ýazgysy: " + str(e))
            else:
                messagebox.showerror(
                    "Näsazlyk", "Girizilen suratda hiç bir ýüz tapylmady. Dogry surat saýlaň!")
        except Exception as e:
            messagebox.showerror(
                "Näsazlyk", "Bir näsazlyk döredi. Ýazgysy: " + str(e))

    def select_image_filedialog(self):
        filetypes = (
            ('PNG file', '*.png *.jpg *.jpeg'),
        )
        self.selected_image_file_path = filedialog.askopenfilename(
            title="Suraty Saýla", filetypes=filetypes)
        self.selected_image = customtkinter.CTkImage(
            Image.open(self.selected_image_file_path), size=(200, 200))
        self.navigation_frame_label = customtkinter.CTkLabel(self.third_frame, text="",
                                                             image=self.selected_image,
                                                             compound="left")
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(
            fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(
            fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.add_staff_frame_button.configure(
            fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")
        self.list_detections_frame_button.configure(
            fg_color=("gray75", "gray25") if name == "list_detections" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()
        if name == "list_detections":
            self.list_detections_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.list_detections_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def list_detection_button_event(self):
        self.select_frame_by_name("list_detections")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.mainloop()
