import cv2
from video_processing.face_recognition import Recognizer
from video_processing.face_spoofing import spoof_detector
from video_processing.face_markers import detect_landmarks
from video_processing.face_orientation import headpose_est
from video_processing.face_orientation import detect_faces, get_face 
from audio_and_oral_movements.microphone import recorder
from audio_and_oral_movements.oral_movement import mouth_open
from cheating import detect_cheating_frame, segment_count # Import necessary functions from the cheating module
from results import plot_main, cheat_count, plot_segments, print_stats # Import necessary functions from the results module
from helper import get_metric_landmarks, convert_bbox, print_faces, print_fps, register_user, combine_transform_matrix, compute_optimal_scale, compute_optimal_rotation, internal_solve_weighted_orthogonal_problem, solve_weighted_orthogonal_problem, extract_square_root, estimate_scale, unproject_xy, move_and_rescale_z, change_handedness, project_xy, cpp_compare # Import necessary functions from the helper module

font = cv2.FONT_HERSHEY_SIMPLEX
pTime = [0]

# Define the register_user function
def register_user(fr, num_pics, skipr):
    # Your implementation of the register_user function goes here
    pass

# Define the print_fps function
def print_fps(frame, pTime):
    # Your implementation of the print_fps function goes here
    pass

# Define the print_faces function
def print_faces(frame, faces):
    # Your implementation of the print_faces function goes here
    pass

# Face recognizer
fr = Recognizer(threshold=0.8)

# Register User
# Assuming fr is an instance of Recognizer class
fr.input_embeddings = register_user(fr, num_pics=5, skipr=False)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('PROCTORING REPORT')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = print_fps(cv2.flip(frame, 1), pTime)

        faces = detect_faces(frame, confidence=0.7)  # Assuming detect_faces function is defined elsewhere
        if faces:
            fr.verify_faces(faces)
            spoof_detector(faces)
            if len(faces) == 1:
                hland = detect_landmarks(frame, faces)  # Assuming detect_landmarks function is defined elsewhere
                if faces[0].landmarks:
                    faces[0].head = headpose_est(frame, faces, hland)  # Assuming headpose_est function is defined elsewhere
                    faces[0].mouth = mouth_open(frame, faces)  # Assuming mouth_open function is defined elsewhere
            frame = print_faces(frame, faces)
        cheat_temp = detect_cheating_frame(faces, frames)  # Assuming detect_cheating_frame function is defined elsewhere
        frames.append(cheat_temp)
        if cheat_temp.cheat > 0:
            cv2.putText(frame, "Please focus on the screen", (15, 105), font, 0.5, (0, 0, 255), 2)
        cv2.imshow('PROCTORING WINDOW', frame)

        recorder()

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    plot_main(frames)  # Assuming plot_main function is defined elsewhere
    segments = segment_count(frames)  # Assuming segment_count function is defined elsewhere
    print_stats(segments)  # Assuming print_stats function is defined elsewhere
    plot_segments(segments)  # Assuming plot_segments function is defined elsewhere

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Your code continues...

    file = open("audio_and_oral_movements/test.txt")  # Student speech file
    data = file.read()
    file.close()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)  # tokenizing sentence
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []

    for w in word_tokens:  # Removing stop words
        if w not in stop_words:
            filtered_sentence.append(w)

        # creating a final file
    f = open('audio_and_oral_movements/final.txt', 'w')
    for ele in filtered_sentence:
        f.write(ele + ' ')
    f.close()

    # checking whether proctor needs to be alerted or not
    file = open("audio_and_oral_movements/paper.txt")  # Question file
    data = file.read()
    file.close()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)  # tokenizing sentence
    filtered_questions = [w for w in word_tokens if not w in stop_words]
    filtered_questions = []

    for w in word_tokens:  # Removing stop words
        if w not in stop_words:
            filtered_questions.append(w)


    def common_member(a, b):
        a_set = set(a)
        b_set = set(b)

        # check length
        if len(a_set.intersection(b_set)) > 0:
            return a_set.intersection(b_set)
        else:
            return []


    comm = common_member(filtered_questions, filtered_sentence)
    print('Number of common elements in recorded voice and question paper:', len(comm))
    print(comm)

