from utils import detector_utils as detector_utils
import cv2
import datetime
import argparse
from random import randint
from kalman_filter import KalmanFilter
# from utils import game_scene_manager

detection_graph, sess = detector_utils.load_inference_graph()


def predict_hand_movement(boxes, num, kalman):
    (left, right, top, bottom) = (boxes[num][1] * im_width, boxes[num][3] * im_width,
                                  boxes[num][0] * im_height, boxes[num][2] * im_height)

    hand_x = left + (right - left) / 2
    hand_y = bottom + (top - bottom) / 2

    return kalman.estimate(hand_x, hand_y)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.2,
                        help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1,
                        help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source', default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int, default=640,
                        help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=480,
                        help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int, default=1,
                        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=5, help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2
    # score circles configuration
    num_circles = 2
    size_circles = 40
    game_time = 10
    circles_exists = 0
    score = 0
    rand_coords = []
    hand_coords = [0]*num_hands_detect

    kalman_filters = []
    for i in range(num_hands_detect):
        kalman_filters.append(KalmanFilter())

    cv2.namedWindow('Hand tracking game', cv2.WINDOW_NORMAL)

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        # detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
        #                                  scores, boxes, im_width, im_height,
        #                                  image_np)

        # predict movement and drav circles
        for i in range(num_hands_detect):
            if scores[i] > args.score_thresh:
                real_coords = detector_utils.get_center_of_box(boxes, i, im_width, im_height)
                cv2.circle(image_np, (real_coords[0], real_coords[1]), 30, (100, 100, 100), 2, 8)
            predicted_coords = predict_hand_movement(boxes, i, kalman_filters[i])
            cv2.circle(image_np, (predicted_coords[0], predicted_coords[1]), 30, (77, 255, 9), 2, 8)
            hand_coords[i] = predicted_coords

        for i in range(num_circles-len(rand_coords)):
            rand_coords.append((randint(size_circles, im_width-size_circles), randint(size_circles, im_height-size_circles)))

        for x in rand_coords:
            cv2.circle(image_np, (x[0], x[1]), size_circles, (255, 0, 0), 4, 8)
            for hand in hand_coords:
                if x[0] - size_circles < hand[0] < x[0] + size_circles \
                        and x[1] - size_circles < hand[1] < x[1] + size_circles:
                    rand_coords.remove(x)
                    score += 1



        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        #cv2.putText(im, "Test", (0, size[1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,), 4)
        #cv2.putText(im, "Test", (0, size[1]), cv2.FONT_HERSHEY_COMPLEX, 2, (255,), 2)

        # Display game time
        time_left = game_time - elapsed_time
        cv2.putText(image_np, 'Time: %.0f s' % time_left, (int(im_width-100), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(image_np, 'Time: %.0f s' % time_left, (int(im_width-100), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Display score
        cv2.putText(image_np, 'Score: %.0f' % score, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(image_np, 'Score: %.0f' % score, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        if time_left < 0.1:
            cv2.putText(image_np, 'End', (int(im_width/2)-100, int(im_height/2)+20), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8)
            cv2.putText(image_np, 'End', (int(im_width/2)-100, int(im_height/2)+20), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)
            while True:
                cv2.imshow('Hand tracking game', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            break

        cv2.imshow('Hand tracking game', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
