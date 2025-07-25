import cv2
import sys
import os
sys.path.append('modules')
import BearTracker

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use

    print("OpenCV version: " + str(major_ver) + "." + str(minor_ver) + "." + str(subminor_ver))

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'BEARTRACKER']
    tracker_type = tracker_types[6]
    #0 'BOOSTING' - slower and more wobly than KCF- looses track on the landings  - does not detect that it looses track
    #1 'MIL' - slower than KCF, and also looses track on landing - does not detect that it looses track
    #2 'KCF' - fast and quite good - but looses track on the landings - does not detect that it looses track
    #3 'TLD' - Slow and looses track - useless
    #4 'MEDIANFLOW' - bad at waves and requires very small init. ROI
    #5 'GOTURN' - does not work - bugs :-(
    #6 'BEARTRACKER'

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'BEARTRACKER':
            tracker = BearTracker.BearTracker(60)

    # Read video
    #video = cv2.VideoCapture("users/bear/full_clips/20170927_15_43_44_GP020491.mp4") #backflip
    #video = cv2.VideoCapture("users/bear/full_clips/20170927_15_42_31_GP020491.mp4") # tail grep
    #video = cv2.VideoCapture("users/bear/full_clips/20170927_15_58_24_GP030491.mp4")  #FS 360
    #video = cv2.VideoCapture("users/bear/full_clips/20170927_15_43_44_GP020491.mp4")  #Backside slide
    #video = cv2.VideoCapture("users/bear/full_clips/20170927_16_00_50_GP040491.mp4")  # backflip fail
    #video = cv2.VideoCapture("users/bear/full_clips/not_bear_20170927_15_42_40_GP020491.mp4")  # far away jump
    video = cv2.VideoCapture(
        os.path.join(
            "C:",
            "git_reps",
            "BearVision",
            "test",
            "users",
            "test_user2",
            "output_video_files",
            "test_user2_20170927_15_58_33.avi",
        )
    )


    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.

    start_frame = 1 #100
    frame_count = 0
    while (frame_count < start_frame):
        ok, frame = video.read()
        frame_count = frame_count + 1
        if not ok:
            print ('Cannot read video file')
            sys.exit()

    # Define an initial bounding box
    bbox = (1, 300, 300, 200) # x1,y1,width,height

    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    print("bbox: " + str(bbox))

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    frame_num = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        frame_num += 1

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            # Tracking failure
            #print("Tracking failure")
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255),2)

        if tracker_type == 'BEARTRACKER':
            tracker.draw(frame)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,170,50),2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,170,50), 2)

        # Display result
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking", frame)
        cv2.moveWindow("Tracking", 10, 10)
        cv2.resizeWindow('Tracking', 1000, 800)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        #if frame_num > 50:
        #    break

    if tracker_type == 'BEARTRACKER':
        tracker.write_state_log_file()