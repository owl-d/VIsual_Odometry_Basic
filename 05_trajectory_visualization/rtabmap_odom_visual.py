import matplotlib.pyplot as plt

position_x = []
position_y = []
position_z = []

orientation_x = []
orientation_y = []
orientation_z = []
orientation_w = []

def process(line):
                
    ### transformation matix 로 pose 얻기 #########################################################
    pose = line.strip().split(",")

    position_x.append(float(pose[5]))       # position
    position_y.append(-float(pose[6]))
    position_z.append(float(pose[7]))
    
    orientation_x.append(float(pose[8]))    # orientation
    orientation_y.append(float(pose[9]))
    orientation_z.append(float(pose[10]))
    orientation_w.append(float(pose[11]))
    ##############################################################################################

idx = 0
with open("/home/doyu_now/08_odom_data.txt", 'r') as f:
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



plt.title('rtabmap odom - scenario 08')

plt.plot(position_y, position_x, 'or')

plt.show()