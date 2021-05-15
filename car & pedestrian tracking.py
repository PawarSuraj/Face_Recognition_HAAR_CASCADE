import cv2

# our image
img_file = "crowded-highway-cars-road.jpg"

# video
video = cv2.VideoCapture('pedestrians.mp4')

# pre-trained car classifier using HAAR
car_tracker_file = 'car_detector.xml'

# predestrian classifier using HAAR
pedestrian_tracker_file= 'haarcascade_fullbody.xml'

# creating car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)

# creating Pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# run forever until car stops
while True:
    # read the current rame
    # read_successful : True/false read was successful or not
    (read_successful, frame) = video.read()

    # if successful
    if read_successful :
        # convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else :
        break

    # detect the cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # detect pedestrians
    pedestrians=pedestrian_tracker.detectMultiScale(grayscaled_frame)


    # draw rectangle over cars
    for (x,y,w,h) in cars :
        cv2.rectangle(frame, (x + 1,y + 1), (x + w,y + h),(0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # draw rectangle over pedestrians
    for (x,y,w,h) in pedestrians :
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255, 255, 255), 2)

    # display every frame in grayscale
    cv2.imshow("The Self Driving Car", frame )

    # 1 milisec gap for every grayscale frame
    key = cv2.waitKey(1)

    # key for quitting
    if key == 81 or key == 113 :
        break
# Release the VideoCapture object
video.release()


'''
# this is an example(Using a single picture) for understanding how to detect a car using HAAR 
# create open-cv image
img = cv2.imread(img_file)

# converting the image to BlacknWhite/graysacle
# it makes the har cascade algorithm faster
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# creating car classifier
# Cascade means series or stack or list of har features in our xml file
car_tracker_file = cv2.CascadeClassifier(car_tracker_file)

# detect the cars
# multi-scale means detect car of any size and shape
cars = car_tracker_file.detectMultiScale(black_n_white)

# draw rectangle around cars
for(x, y, w, h) in cars:
    cv2.rectangle(black_n_white, (x, y), (x+w, y+h), (0, 0, 225), 2)

# display the image with faces spotted
cv2.imshow("the highway car image", black_n_white)
# if cv2.imshow doesnt work run "pip install opencv-contrib-python" on cmd

# custom key to close window
cv2.waitKey()

print('code completed')
'''