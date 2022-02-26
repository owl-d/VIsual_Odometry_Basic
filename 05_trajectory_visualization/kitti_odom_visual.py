import matplotlib.pyplot as plt


x_list = []
y_list = []

def process(line):
                
    ### transformation matix 로 궤적 얻기 ##############################################################
    pose = line.strip().split()

    x_list.append(float(pose[3]))  # position
    y_list.append(float(pose[11]))

    ##################################################################################################

file_path = '/media/doyu_now/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses/06.txt'
with open(file_path, 'r') as f:
    while True:

        line = f.readline()
        if not line:
            print("End Of File")
            break
        
        process(line)

plt.plot(x_list, y_list, 'ob')

plt.title('kitti odom - scenario 04')
plt.show()