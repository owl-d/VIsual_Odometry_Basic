import matplotlib.pyplot as plt


def process(line):
                
    ### transformation matix 로 pose 계산 : 6DoF ##############################################################
    pose = line.strip().split()

    x = pose[3] #position
    y = pose[7]
    ##########################################################################################################

    plt.plot(x, y, 'ob')

file_path = '/media/doyu_now/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses/04.txt'
with open(file_path, 'r') as f:
    while True:

        line = f.readline()
        if not line:
            print("End Of File")
            break
        
        process(line)


plt.title('kitti odom - scenario 04')
plt.show()