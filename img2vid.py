import glob
import cv2 as cv
import os

"""
For downloading the original Front center camera images for each city  please refer to https://www.a2d2.audi/a2d2/en/download.html 

Please Change the right location name,
the available locations are "Gaimersheim", "Munich" and "Ingolstadt"
set the date_of_recording, time_of_recording and type of camera used 

"""


def main(location):
    # define location of the downloaded camera images.jpeg below, date_of_recording, time_of_recording and camera type
    date_of_recording = "20180810"  # date in format YYYYMMDD
    time_of_recording = "150607"  # time in format HHMMSS
    camera = "cam_front_left"

    cam_data_path = os.path.join("data/Camera_data/", location, "Front_Camera_images", camera)
    cam_images_files_png = glob.glob(cam_data_path + "/*.png")
    cam_images_files_png.sort()

    img_0 = cv.imread(cam_images_files_png[0])
    height, width, layers = img_0.shape
    size = (width, height)

    # please define a file path for saving the video
    out = cv.VideoWriter('Camera_data', location,
                         'Front_Camera_Video/' + "_" + camera + date_of_recording + "_" + time_of_recording + ".mp4",
                         cv.VideoWriter_fourcc(*'MJPG'), 30, size)

    for filename in cam_images_files_png:
        img = cv.imread(filename)
        out.write(img)

    out.release()


if __name__ == "__main__":
    location = "Gaimersheim"
    main(location)
