import matplotlib.pyplot as plt


def process(line):
                
    ### transformation matix 로 pose 얻기 #########################################################
    pose = line.strip().split(",")

    position_x = pose[5] #position
    position_y = pose[6]
    position_z = pose[7]
    orientation_x = pose[8] #orientation
    orientation_y = pose[9]
    orientation_z = pose[10]
    orientation_w = pose[11]
    ##############################################################################################

    plt.plot(position_x, position_y, 'or')



idx = 0
with open("/home/doyu_now/04_odom_data.txt", 'r') as f:
    while True:

        line = f.readline()
        if not line:
            print("End Of File")
            break
        
        elif idx == 0:
            idx += 1

        else:
            process(line)
            idx += 1


plt.title('rtabmap odom - scenario 04')
plt.show()