import datetime
import math
import os
import pickle
import time

import cv2
import dlib
import face_recognition
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
# import mpld3
import matplotlib.pyplot as plty

import numpy as np
import pandas as pd
import seaborn as sns
from attendance_system_facial_recognition.settings import BASE_DIR
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count
from django.shortcuts import render, redirect
from django_pandas.io import read_frame
from face_recognition.face_recognition_cli import image_files_in_folder
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils.video import VideoStream
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from users.models import Present, Time

from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2

mpl.use('Agg')


# utility functions:
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True

    return False


def create_dataset(username):
    id = username
    directory = 'face_recognition_data/training_dataset/{}/'.format(id)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize face detector and aligner
    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=96)

    # Start video stream
    print("[INFO] Initializing Video stream")
    vs = VideoStream(src=0).start()
    vs.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vs.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(2.0)  # Allow the camera to warm up

    # Setup variables
    sampleNum = 0
    total_images_per_instruction = 100
    instructions = [
        "PLEASE LOOK AT THE CAMERA",
        "PLEASE SMILE",
        "OPEN YOUR EYES WIDE, THANK YOU"
    ]
    instruction_index = 0
    instruction_count = 0
    instruction_change_interval = total_images_per_instruction

    while True:
        frame = vs.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        # Display current instruction
        instruction_text = instructions[instruction_index]
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame, gray_frame, face)
            
            # Only save images if the face is detected
            if face is not None:
                sampleNum += 1
                cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', face_aligned)
                face_aligned = imutils.resize(face_aligned, width=400)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.waitKey(50)
        
        # Display the frame with instruction
        cv2.imshow("Add Images", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Check if we have captured enough images for the current instruction
        if instruction_count >= instruction_change_interval:
            instruction_index = (instruction_index + 1) % len(instructions)
            instruction_count = 0  # Reset counter for the next instruction
            print(f"[INFO] Changing to next instruction: {instructions[instruction_index]}")
            time.sleep(2.0)  # Short pause before changing instruction

        instruction_count += 1

        # Stop after capturing enough images in total
        if sampleNum >= len(instructions) * total_images_per_instruction:
            break

    # Cleanup
    vs.stop()
    cv2.destroyAllWindows()



import numpy as np
import face_recognition

def predict(face_aligned, svc, threshold=0.70):
    """
    Predict the identity of the aligned face using the trained SVC model.

    Parameters:
    - face_aligned: The aligned face image.
    - svc: Trained Support Vector Classifier.
    - threshold: Probability threshold to classify a face as unknown.

    Returns:
    - Tuple of (predicted_label, probability).
    """
    try:
        # Ensure face is in RGB format for face_recognition
        face_aligned_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        
        # Detect face locations and encode faces
        face_locations = face_recognition.face_locations(face_aligned_rgb)
        face_encodings = face_recognition.face_encodings(face_aligned_rgb, known_face_locations=face_locations)

        if not face_encodings:
            # No faces detected
            return ([-1], [0.0])

        # Predict probabilities for detected face encodings
        prob = svc.predict_proba(face_encodings)
        max_prob = np.max(prob, axis=1)  # Find the maximum probability
        max_prob_idx = np.argmax(prob, axis=1)  # Find the index of the max probability

        # If the highest probability is below the threshold, classify as unknown
        if max_prob[0] < threshold:
            return ([-1], max_prob[0])  # Return -1 for unknown with its probability

        # Otherwise, return the predicted class index and probability
        return (max_prob_idx[0], max_prob[0])
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ([-1], [0.0])

def vizualize_Data(embedded, targets, ):
    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1));
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
    plt.close()


def update_attendance_in_db_in(present):
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        try:
            qs = Present.objects.get(user=user, date=today)
        except:
            qs = None

        if qs is None:
            if present[person] == True:
                a = Present(user=user, date=today, present=True)
                a.save()
            else:
                a = Present(user=user, date=today, present=False)
                a.save()
        else:
            if present[person] == True:
                qs.present = True
                qs.save(update_fields=['present'])
        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=False)
            a.save()


def update_attendance_in_db_out(present):
    """
    Updates attendance records for users marking their 'out' status.

    Args:
        present (dict): A dictionary with usernames as keys and attendance status (True/False) as values.
    """
    today = datetime.date.today()

    for username, is_present in present.items():
        if not is_present:
            continue
        
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            print(f"User '{username}' does not exist.")
            continue
        
        # Record the 'out' time
        current_time = datetime.datetime.now()
        Time.objects.create(user=user, date=today, time=current_time, out=True)

def check_validity_times(times_all):
    if (len(times_all) > 0):
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False)
    times_out = times_all.filter(out=True)
    if (len(times_in) != len(times_out)):
        sign = True
    break_hourss = 0
    if (sign == True):
        check = False
        break_hourss = 0
        return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time

    for obj in times_all:
        curr = obj.out
        if (curr == prev):
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if (curr == False):
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            break_time = ((to - ti).total_seconds()) / 3600
            break_hourss += break_time


        else:
            prev_time = obj.time

        prev = curr

    return (True, break_hourss)


def convert_hours_to_hours_mins(hours):
    h = int(hours)
    hours -= h
    m = hours * 60
    m = math.ceil(m)
    return str(str(h) + " hrs " + str(m) + "  mins")


# used
def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        times_in = time_qs.filter(date=date).filter(out=False).order_by('time')
        times_out = time_qs.filter(date=date).filter(out=True).order_by('time')
        times_all = time_qs.filter(date=date).order_by('time')
        obj.time_in = None
        obj.time_out = None
        obj.hours = 0
        obj.break_hours = 0
        if (len(times_in) > 0):
            obj.time_in = times_in.first().time

        if (len(times_out) > 0):
            obj.time_out = times_out.last().time

        if (obj.time_in is not None and obj.time_out is not None):
            ti = obj.time_in
            to = obj.time_out
            hours = ((to - ti).total_seconds()) / 3600
            obj.hours = hours


        else:
            obj.hours = 0

        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss


        else:
            obj.break_hours = 0

        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)

    df["hours"] = df_hours
    df["break_hours"] = df_break_hours

    print(df)

    sns.barplot(data=df, x='date', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    if (admin):
        plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
        plt.close()
    else:
        plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
        plt.close()
    return qs


# used
def hours_vs_employee_given_date(present_qs, time_qs):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []
    qs = present_qs

    for obj in qs:
        user = obj.user
        times_in = time_qs.filter(user=user).filter(out=False)
        times_out = time_qs.filter(user=user).filter(out=True)
        times_all = time_qs.filter(user=user)
        obj.time_in = None
        obj.time_out = None
        obj.hours = 0
        obj.hours = 0
        if (len(times_in) > 0):
            obj.time_in = times_in.first().time
        if (len(times_out) > 0):
            obj.time_out = times_out.last().time
        if (obj.time_in is not None and obj.time_out is not None):
            ti = obj.time_in
            to = obj.time_out
            hours = ((to - ti).total_seconds()) / 3600
            obj.hours = hours
        else:
            obj.hours = 0
        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss


        else:
            obj.break_hours = 0

        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)
    df['hours'] = df_hours
    df['username'] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x='username', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
    plt.close()
    return qs


def total_number_employees():
    qs = User.objects.all()
    return (len(qs) - 1)


# -1 to account for admin


def employees_present_today():
    today = datetime.date.today()
    qs = Present.objects.filter(date=today).filter(present=True)
    return len(qs)


# used
def this_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs))

    while (cnt < 5):

        date = str(monday_of_this_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if (str_dates.count(date)) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["Number of employees"] = emp_cnt_all

    sns.lineplot(data=df, x='date', y='Number of employees')
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
    plt.close()


# used
def last_week_emp_count_vs_date():
    today = datetime.date.today()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
    str_dates = []
    emp_count = []

    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs))

    while (cnt < 5):

        date = str(monday_of_last_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if (str_dates.count(date)) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])

        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["emp_count"] = emp_cnt_all

    sns.lineplot(data=df, x='date', y='emp_count')
    plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
    plt.close()


# Create your views here.
def home(request):
    return render(request, 'recognition/home.html')


@login_required
def dashboard(request):
    if (request.user.username == 'admin'):
        print("admin")
        return render(request, 'recognition/admin_dashboard.html')
    else:
        print("not admin")

        return render(request, 'recognition/employee_dashboard.html')


@login_required
def add_photos(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    if request.method == 'POST':
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get('username')
        if username_present(username):
            create_dataset(username)
            messages.success(request, f'Dataset Created')
            return redirect('add-photos')
        else:
            messages.warning(request, f'No such username found. Please register employee first.')
            return redirect('dashboard')


    else:

        form = usernameForm()
        return render(request, 'recognition/add_photos.html', {'form': form})

def mark_your_attendance(request):
    """
    Detects faces, displays live video stream with square boxes, names, and scores.
    Marks attendance when 'q' is pressed.
    """
    # Initialize face detection and recognition components
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)

    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    # Initialize attendance tracking
    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    present = {encoder.inverse_transform([i])[0]: False for i in range(no_of_faces)}

    # Start video stream
    vs = VideoStream(src=0).start()
    time.sleep(1.0)  # Allow the camera to warm up slightly for better responsiveness

    # Set probability threshold and initialize frame count
    probability_threshold = 0.7
    captured = False
    frame_count = 0
    detection_interval = 5  # Perform face detection every 5 frames

    while not captured:
        # Read the frame from the video stream
        frame = vs.read()
        frame = imutils.resize(frame, width=600)  # Reduce frame size for faster processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only perform face detection every few frames to reduce lag
        if frame_count % detection_interval == 0:
            faces = detector(gray_frame, 0)
        
        # Process each detected face
        for face in faces:
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            size = max(w, h)
            x_end = x + size
            y_end = y + size

            face_aligned = fa.align(frame, gray_frame, face)
            (pred, prob) = predict(face_aligned, svc)

            # Determine box color based on threshold
            prob_value = prob[0] if isinstance(prob, list) else prob
            box_color = (0, 255, 0) if prob_value >= probability_threshold else (0, 0, 255)

            # Draw square around the face and display name and score
            cv2.rectangle(frame, (x, y), (x_end, y_end), box_color, 2)
            person_name = encoder.inverse_transform(np.ravel([pred]))[0] if pred != [-1] else "Unknown"
            prob_str = f"{prob_value:.2f}"
            cv2.putText(frame, f"{person_name} {prob_str}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Display the live video stream
        cv2.imshow("Mark Attendance - Press q to capture", frame)
        frame_count += 1  # Increment frame counter

        # Check if 'q' is pressed to capture and mark attendance
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if faces:
                face = faces[0]  # Take the first detected face
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                size = max(w, h)
                x_end = x + size
                y_end = y + size

                face_aligned = fa.align(frame, gray_frame, face)
                (pred, prob) = predict(face_aligned, svc)

                # Check if the face is recognized
                if pred != [-1]:
                    person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                    present[person_name] = True
                    text = f"{person_name.title()} Your Attendance-In is marked"
                else:
                    person_name = "Unknown"
                    text = "We do not know you!"

                # Display the result window with the appropriate message
                background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (background_image.shape[1] - text_size[0]) // 2
                text_y = (background_image.shape[0] + text_size[1]) // 3
                cv2.putText(background_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Face Detected", background_image)
                cv2.waitKey(3000)  # Display for 3 seconds to reduce lag
                cv2.destroyWindow("Face Detected")
                
                captured = True
            else:
                # Handle case where no face is detected
                background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
                text = "No face detected. Unable to mark attendance."
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (background_image.shape[1] - text_size[0]) // 2
                text_y = (background_image.shape[0] + text_size[1]) // 3
                cv2.putText(background_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Face Detected", background_image)
                cv2.waitKey(3000)  # Display for 3 seconds to reduce lag
                cv2.destroyWindow("Face Detected")
                
                captured = True

    vs.stop()
    cv2.destroyAllWindows()
    update_attendance_in_db_in(present)  # Call this function to update the attendance in the database
    return redirect('home')

# def mark_your_attendance_out(request):
#     """
#     Detects faces, marks attendance as 'out' for recognized faces, and updates the database.
#     """
#     # Initialize face detection and recognition components
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
#     svc_save_path = "face_recognition_data/svc.sav"

#     with open(svc_save_path, 'rb') as f:
#         svc = pickle.load(f)
        
#     fa = FaceAligner(predictor, desiredFaceWidth=96)
#     encoder = LabelEncoder()
#     encoder.classes_ = np.load('face_recognition_data/classes.npy')

#     # Start video stream
#     vs = VideoStream(src=0).start()
#     time.sleep(1.0)  # Allow the camera to warm up slightly for better responsiveness

#     face_captured = False
#     captured_image = None
#     status_text = ""
#     present = {}  # Initialize the dictionary to keep track of attendance status
#     frame_count = 0
#     detection_interval = 5  # Perform face detection every 5 frames

#     while not face_captured:
#         frame = vs.read()
#         frame = imutils.resize(frame, width=600)  # Reduce frame size for faster processing
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Only perform face detection every few frames to reduce lag
#         if frame_count % detection_interval == 0:
#             faces = detector(gray_frame, 0)

#         # Process each detected face
#         for face in faces:
#             (x, y, w, h) = face_utils.rect_to_bb(face)
#             face_aligned = fa.align(frame, gray_frame, face)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
#             (pred, prob) = predict(face_aligned, svc)

#             if pred != [-1]:
#                 person_name = encoder.inverse_transform(np.ravel([pred]))[0]
#                 prob_str = f"{prob[0]:.2f}" if isinstance(prob, list) else f"{prob:.2f}"
#                 cv2.putText(frame, f"{person_name} {prob_str}", (x + 6, y + h - 6), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
#                 status_text = f"{person_name} - Attendance Marked Out"
#                 captured_image = frame.copy()
#                 present[person_name] = True
#             else:
#                 person_name = "Unknown"
#                 prob_str = f"{prob[0]:.2f}" if isinstance(prob, list) else f"{prob:.2f}"
#                 cv2.putText(frame, f"{person_name} {prob_str}", (x + 6, y + h - 6), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
#                 status_text = "We do not know you!"
#                 captured_image = frame.copy()
#                 present[person_name] = False

#         # Display the live video stream
#         cv2.imshow("Mark Attendance - Out - Press q to capture", frame)
#         frame_count += 1  # Increment frame counter

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             if captured_image is not None:
#                 face_captured = True
#             else:
#                 print("No face detected. Press 'q' again after a face appears.")

#     if captured_image is not None:
#         # Display the status message on a background image
#         background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text_color = (255, 255, 255) if "Marked Out" in status_text else (255, 0, 0)  # White for marked out, red for unknown
#         text_size = cv2.getTextSize(status_text, font, 1, 2)[0]
#         text_x = (background_image.shape[1] - text_size[0]) // 2
#         text_y = (background_image.shape[0] + text_size[1]) // 2
#         cv2.putText(background_image, status_text, (text_x, text_y), font, 1, text_color, 2, cv2.LINE_AA)
        
#         # Display the captured image with the status
#         cv2.imshow("Detected Face Status", background_image)
#         cv2.waitKey(3000)  # Show the image for 3 seconds to reduce lag
#         cv2.destroyWindow("Detected Face Status")

#     # Stop video stream and close windows
#     vs.stop()
#     cv2.destroyAllWindows()

#     # Update attendance in the database
#     update_attendance_in_db_out(present)  
#     return redirect('home')

# def mark_your_attendance(request):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
#     svc_save_path = "face_recognition_data/svc.sav"

#     with open(svc_save_path, 'rb') as f:
#         svc = pickle.load(f)
        
#     fa = FaceAligner(predictor, desiredFaceWidth=96)
#     encoder = LabelEncoder()
#     encoder.classes_ = np.load('face_recognition_data/classes.npy')

#     faces_encodings = np.zeros((1, 128))
#     no_of_faces = len(svc.predict_proba(faces_encodings)[0])
#     count = {encoder.inverse_transform([i])[0]: 0 for i in range(no_of_faces)}
#     present = {encoder.inverse_transform([i])[0]: False for i in range(no_of_faces)}
#     start = {}

#     # Start video stream
#     vs = VideoStream(src=0).start()

#     captured = False

#     while not captured:
#         frame = vs.read()
#         frame = imutils.resize(frame, width=800)
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray_frame, 0)

#         # Display live video stream
#         cv2.imshow("Mark Attendance - Press q to capture", frame)

#         # Check if 'q' is pressed
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             if faces:
#                 # Process the first detected face
#                 face = faces[0]
#                 (x, y, w, h) = (face.left(), face.top(), face.right(), face.bottom())
#                 face_aligned = fa.align(frame, gray_frame, face)
                
#                 # Predict the face
#                 (pred, prob) = predict(face_aligned, svc)

#                 if pred != [-1]:
#                     person_name = encoder.inverse_transform(np.ravel([pred]))[0]
#                     present[person_name] = True
#                     background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
#                     text = f"{person_name.title()} Your Attendance-In is marked"
#                 else:
#                     person_name = "unknown"
#                     background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
#                     text = "We do not know you!"

#                 text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
#                 text_x = (background_image.shape[1] - text_size[0]) // 2
#                 text_y = (background_image.shape[0] + text_size[1]) // 3
#                 cv2.putText(background_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                
#                 # Show result window
#                 cv2.imshow("Face Detected", background_image)
#                 cv2.waitKey(5000)
#                 cv2.destroyWindow("Face Detected")

#                 captured = True
#             else:
#                 # Create a background image for no face detected
#                 background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
#                 text = "No face detected Unable to mark attendance"
#                 text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
#                 text_x = (background_image.shape[1] - text_size[0]) // 2
#                 text_y = (background_image.shape[0] + text_size[1]) // 3
#                 cv2.putText(background_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
#                 # Show result window
#                 cv2.imshow("Face Detected", background_image)
#                 cv2.waitKey(5000)
#                 cv2.destroyWindow("Face Detected")
                
#                 captured = True

#     vs.stop()
#     cv2.destroyAllWindows()
#     update_attendance_in_db_in(present)  # Call this function to update the attendance in the database
#     return redirect('home')


def mark_your_attendance_out(request):
    """
    Detects faces, marks attendance as 'out' for recognized faces, and updates the database.
    """
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
        svc_save_path = "face_recognition_data/svc.sav"

        with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
            
        fa = FaceAligner(predictor, desiredFaceWidth=96)
        encoder = LabelEncoder()
        encoder.classes_ = np.load('face_recognition_data/classes.npy')

        vs = VideoStream(src=0).start()
        time.sleep(2.0)  # Allow camera to warm up

        face_captured = False
        captured_image = None
        status_text = ""
        present = {}  # Initialize the dictionary to keep track of attendance status

        while not face_captured:
            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)

            for face in faces:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                face_aligned = fa.align(frame, gray_frame, face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                (pred, prob) = predict(face_aligned, svc)

                if pred != [-1]:
                    person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                    prob_str = f"{prob[0]:.2f}" if isinstance(prob, list) else f"{prob:.2f}"
                    cv2.putText(frame, f"{person_name} {prob_str}", (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    status_text = f"{person_name} - Attendance Marked Out"
                    captured_image = frame.copy()
                    present[person_name] = True
                else:
                    person_name = "Unknown"
                    prob_str = f"{prob[0]:.2f}" if isinstance(prob, list) else f"{prob:.2f}"
                    cv2.putText(frame, f"{person_name} {prob_str}", (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    status_text = "We do not know you!"
                    captured_image = frame.copy()
                    present[person_name] = False

            cv2.imshow("Mark Attendance - Out - Press q to capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if captured_image is not None:
                    background_image = np.full((150, 700, 3), (0, 0, 0), dtype=np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_color = (255, 255, 255) if "Marked Out" in status_text else (255, 0, 0)  # White for marked out, red for unknown
                    text_size = cv2.getTextSize(status_text, font, 1, 2)[0]
                    text_x = (background_image.shape[1] - text_size[0]) // 2
                    text_y = (background_image.shape[0] + text_size[1]) // 2
                    cv2.putText(background_image, status_text, (text_x, text_y), font, 1, text_color, 2, cv2.LINE_AA)
                    
                    # Display the captured image with the status
                    cv2.imshow("Detected Face Status", background_image)
                    cv2.waitKey(5000)  # Show the image for 5 seconds
                    cv2.destroyWindow("Detected Face Status")
                    face_captured = True
                else:
                    print("No face detected. Press 'q' again after a face appears.")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        vs.stop()
        cv2.destroyAllWindows()
        try:
            update_attendance_in_db_out(present)  # Update attendance in the database
        except Exception as e:
            print(f"An error occurred while updating the database: {e}")
        return redirect('home')


# def mark_your_attendance_out(request):
#     """
#     Detects faces, marks attendance as 'out' for recognized faces, and updates the database.
#     """
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')  # Adjust path if needed
#     svc_save_path = "face_recognition_data/svc.sav"

#     with open(svc_save_path, 'rb') as f:
#         svc = pickle.load(f)

#     fa = FaceAligner(predictor, desiredFaceWidth=96)
#     encoder = LabelEncoder()
#     encoder.classes_ = np.load('face_recognition_data/classes.npy')

#     # Initialize attendance tracking dictionaries
#     faces_encodings = np.zeros((1, 128))
#     no_of_faces = len(svc.predict_proba(faces_encodings)[0])
#     count = {encoder.inverse_transform([i])[0]: 0 for i in range(no_of_faces)}
#     present = {encoder.inverse_transform([i])[0]: False for i in range(no_of_faces)}
#     log_time = {}

#     vs = VideoStream(src=0).start()
#     sampleNum = 0

#     while True:
#         frame = vs.read()
#         frame = imutils.resize(frame, width=800)
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray_frame, 0)

#         for face in faces:
#             (x, y, w, h) = face_utils.rect_to_bb(face)
#             face_aligned = fa.align(frame, gray_frame, face)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
#             (pred, prob) = predict(face_aligned, svc)

#             if pred != [-1]:
#                 person_name = encoder.inverse_transform(np.ravel([pred]))[0]
#                 pred = person_name

#                 if count[pred] == 0:
#                     count[pred] = 1

#                 if count[pred] == 4:
#                     count[pred] = 0
#                 else:
#                     present[pred] = True
#                     log_time[pred] = datetime.datetime.now()
#                     count[pred] += 1
#                     print(pred, present[pred], count[pred])

#                 cv2.putText(frame, f"{person_name} {prob}", (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             else:
#                 person_name = "unknown"
#                 cv2.putText(frame, person_name, (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#         # Show the video frame
#         cv2.imshow("Mark Attendance - Out - Press q to exit", frame)
#         key = cv2.waitKey(50) & 0xFF
#         if key == ord("q"):
#             break

#     # Stop the video stream and close windows
#     vs.stop()
#     cv2.destroyAllWindows()

#     # Call the update function with the attendance data
#     update_attendance_in_db_out(present)
#     return redirect('home')

    

@login_required
def train(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')

    training_dir = 'face_recognition_data/training_dataset'

    count = 0
    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            count += 1

    X = []
    y = []
    i = 0

    for person_name in os.listdir(training_dir):
        print(str(person_name))
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                X.append((face_recognition.face_encodings(image)[0]).tolist())

                y.append(person_name)
                i += 1
            except:
                print("removed")
                os.remove(imagefile)

    targets = np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    np.save('face_recognition_data/classes.npy', encoder.classes_)
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X1, y)
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f)

    vizualize_Data(X1, targets)

    messages.success(request, f'Training Complete.')

    return render(request, "recognition/train.html")


@login_required
def not_authorised(request):
    return render(request, 'recognition/not_authorised.html')


@login_required
def view_attendance_home(request):
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_emp_count_vs_date()
    last_week_emp_count_vs_date()
    return render(request, "recognition/view_attendance_home.html",
                  {'total_num_of_emp': total_num_of_emp, 'emp_present_today': emp_present_today})


@login_required
def view_attendance_date(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None

    if request.method == 'POST':
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data.get('date')
            print("date:" + str(date))
            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)
            if (len(time_qs) > 0 or len(present_qs) > 0):
                qs = hours_vs_employee_given_date(present_qs, time_qs)

                return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})
            else:
                messages.warning(request, f'No records for selected date.')
                return redirect('view-attendance-date')








    else:

        form = DateForm()
        return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})


@login_required
def view_attendance_employee(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    time_qs = None
    present_qs = None
    qs = None

    if request.method == 'POST':
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            if username_present(username):

                u = User.objects.get(username=username)

                time_qs = Time.objects.filter(user=u)
                present_qs = Present.objects.filter(user=u)
                date_from = form.cleaned_data.get('date_from')
                date_to = form.cleaned_data.get('date_to')

                if date_to < date_from:
                    messages.warning(request, f'Invalid date selection.')
                    return redirect('view-attendance-employee')
                else:

                    time_qs = time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
                    present_qs = present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')

                    if (len(time_qs) > 0 or len(present_qs) > 0):
                        qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})
                    else:
                        # print("inside qs is None")
                        messages.warning(request, f'No records for selected duration.')
                        return redirect('view-attendance-employee')






            else:
                print("invalid username")
                messages.warning(request, f'No such username found.')
                return redirect('view-attendance-employee')


    else:

        form = UsernameAndDateForm()
        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})


@login_required
def view_my_attendance_employee_login(request):
    if request.user.username == 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None
    if request.method == 'POST':
        form = DateForm_2(request.POST)
        if form.is_valid():
            u = request.user
            time_qs = Time.objects.filter(user=u)
            present_qs = Present.objects.filter(user=u)
            date_from = form.cleaned_data.get('date_from')
            date_to = form.cleaned_data.get('date_to')
            if date_to < date_from:
                messages.warning(request, f'Invalid date selection.')
                return redirect('view-my-attendance-employee-login')
            else:

                time_qs = time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
                present_qs = present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')

                if (len(time_qs) > 0 or len(present_qs) > 0):
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
                    return render(request, 'recognition/view_my_attendance_employee_login.html',
                                  {'form': form, 'qs': qs})
                else:

                    messages.warning(request, f'No records for selected duration.')
                    return redirect('view-my-attendance-employee-login')
    else:

        form = DateForm_2()
        return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})


@login_required
def logout(request):
    return redirect('home')
