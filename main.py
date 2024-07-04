import hydra

from motion_tracker_custom import MotionTrackerCustom
from motion_tracker_cv2 import MotionTrackerCV2


@hydra.main(version_base=None, config_path="conf", config_name="app_config")
def main(config) -> None:
    m1 = MotionTrackerCV2(config)
    m2 = MotionTrackerCustom(config)
    m1.print_video()
    m2.print_video()


if __name__ == "__main__":
    main()
