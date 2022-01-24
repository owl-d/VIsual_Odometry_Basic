import visual_odometry_ORB_OptFlow_KITTI as vo

from matplotlib import pyplot as plt

VO_KITTI = vo.mono_VO_ORBFlow_KITTI(718.856, 718.856, 718.856, 607.1928, 185.2157,
                                    './02_Monocular_VO_Pipeline_Implement/KITTI_dataset/sequences/00/image_2/',
                                    './02_Monocular_VO_Pipeline_Implement/KITTI_dataset/poses/00.txt')

idx = 0
count = 0
while True:

    VO_KITTI.img_buffer_feature_tracking(disp_img=True)

    if idx >= 3:
        if VO_KITTI.frame_Skip() == False:

            VO_KITTI.geometric_change_calc()
            VO_KITTI.img_common3Dcloud_triangulate()
            VO_KITTI.pose_estimate()
            VO_KITTI.update()

            #Draw the trajectory
            plt.title('KITTI Dataset - Monocular Visual Odometry (Relative Translation Scaling)\n[ORB-based Optical Flow]')
            plt.plot(VO_KITTI.pose_T[0][0], VO_KITTI.pose_T[2][0], 'ro')

            #Draw the groundtruth
            plt.plot(VO_KITTI.ground_truth_T[VO_KITTI.dataset_current_idx-1][0], VO_KITTI.ground_truth_T[VO_KITTI.dataset_current_idx-1][2], 'bo')

            plt.pause(0.000001)
            plt.show(block=False)

        else:
            print('[FRAME SKIPPED] : Camera is stationary / No need to accumulate pose data')
    
    else:
        idx += 1

    print('--------------------------------------------------------')
    count += 1
    print('count : ', count)