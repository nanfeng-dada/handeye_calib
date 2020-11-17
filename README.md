# handeye_calibrate
handeye_calibrate

- 代码适用于eye to hand手眼标定，博客https://blog.csdn.net/qq_34782535/article/details/109181362

  输入：棋盘格图片和机器人末端位姿
  
  输出：机器人坐标系下相机的位姿

- 说明：

  1. 本代码验证了opencv手眼标定用于eye to hand的方式
  
  2.机器人末端为x y z rx ry rz表示，存于robot.txt中；单位mm、 rad
  
  3.输出矩阵平移向量单位m
