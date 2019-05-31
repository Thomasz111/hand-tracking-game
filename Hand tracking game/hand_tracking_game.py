from utils import detector_utils as detector_utils
import cv2
import datetime
import argparse
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

    kalman_filters = []
    for i in range(num_hands_detect):
        kalman_filters.append(KalmanFilter())

    cv2.namedWindow('Hand tracking game', cv2.WINDOW_NORMAL)
    # game_scene = game_scene_manager.GameScene(640, 480, 'Hand tracking game')

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
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        # predict movement and drav circles
        # game_scene.clear_scene()
        for i in range(num_hands_detect):
            predicted_coords = predict_hand_movement(boxes, i, kalman_filters[i])
            # game_scene.write_circle(predicted_coords[0], predicted_coords[1], 20)
            cv2.circle(image_np, (predicted_coords[0], predicted_coords[1]), 20, (77, 255, 9), 2, 8)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if args.display > 0:
            # Display FPS on frame
            if args.fps > 0:
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Hand tracking game', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            # game_scene.show_scene()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
