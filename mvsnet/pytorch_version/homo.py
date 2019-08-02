import torch
def get_pixel_grids(height, width):
    # texture coordinate
    x_linspace = torch.linspace(0.5, width- 0.5, width)
    y_linspace = torch.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
    x_coordinates = torch.reshape(x_coordinates, [-1])
    y_coordinates = torch.reshape(y_coordinates, [-1])
    ones = torch.ones(x_coordinates.shape)
    indices_grid = torch.cat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid
def reprojection(input_image,left_cam,right_cam,depth_map):
    """

    :param input_image:
    :param left_cam:
    :param right_cam:
    :param depth_map: b,h,w,1
    :return:
    """
    input_image=input_image.permute(0,2,3,1)#b,h,w,c
    image_shape = input_image.shape
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = torch.unsqueeze(pixel_grids, 0)
    pixel_grids = torch.cat([pixel_grids]*batch_size, 0)
    pixel_grids = torch.reshape(pixel_grids, (batch_size, 3, -1))#b,3,[hxw]
    depth_flatten=torch.reshape(depth_map,(batch_size,1,-1))#b,1,[hxw]
    pixel_grids=pixel_grids*depth_flatten
    R_left =left_cam[:,0,:3,:3]# tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    R_right =right_cam[:,0,:3,:3]#tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    t_left = left_cam[:,0,:3,3]#tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    t_right = right_cam[:,0,:3,3]#tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    K_left = left_cam[:,1,:3,:3]#tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_right = right_cam[:,1,:3,:3]#tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_left_inv = torch.inverse(K_left)
    # K_right=tf.squeeze(K_right,1)
    R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
    R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
    c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
    c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
    c_relative = tf.subtract(c_right, c_left)#b,3,1
    middle_mat1 = tf.matmul(R_left_trans, K_left_inv)#b,3,3
    middle_mat2 = tf.squeeze(tf.matmul(K_right, R_right),1)  # b,3,3
    pixel_grids=tf.matmul(middle_mat1,pixel_grids)#b,3,[h,w]
    pixel_grids=tf.subtract(pixel_grids,c_relative)
    pixel_grids=tf.matmul(middle_mat2,pixel_grids)
    grids_div=tf.slice(pixel_grids,[0,2,0],[-1,1,-1])#b,1,[h,w]
    grids_zero_add = tf.cast(tf.equal(grids_div, 0.0), dtype='float32') * 1e-7  # handle div 0
    grids_div = grids_div + grids_zero_add
    grids_div = tf.tile(grids_div, [1, 2, 1])
    grids_affine=tf.slice(pixel_grids,[0,0,0],[-1,2,-1])
    grids_inv_warped = tf.div(grids_affine, grids_div)
    x_warped, y_warped = tf.unstack(grids_inv_warped, axis=1)
    x_warped_flatten = tf.reshape(x_warped, [-1])
    y_warped_flatten = tf.reshape(y_warped, [-1])
    warped_image = interpolate(input_image, x_warped_flatten, y_warped_flatten)
    warped_image = tf.reshape(warped_image, shape=image_shape, name='warped_feature')
    mask=tf.reshape((x_warped>=0.0) & (x_warped<tf.cast(width,tf.float32)) & (y_warped>=0.0) &( y_warped<tf.cast(height,tf.float32)),
                    [batch_size,height,width,1])
    return warped_image,mask
def PQ(left_cam,right_cam,image_shape):
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]
    pixel_grids = get_pixel_grids(height, width)
    pixel_grids = tf.expand_dims(pixel_grids, 0)
    pixel_grids = tf.tile(pixel_grids, [batch_size, 1])
    pixel_grids = tf.reshape(pixel_grids, (batch_size, 3, -1))  # b,3,[hxw]

    R_left = tf.slice(left_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    R_right = tf.slice(right_cam, [0, 0, 0, 0], [-1, 1, 3, 3])
    t_left = tf.slice(left_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    t_right = tf.slice(right_cam, [0, 0, 0, 3], [-1, 1, 3, 1])
    K_left = tf.slice(left_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_right = tf.slice(right_cam, [0, 1, 0, 0], [-1, 1, 3, 3])
    K_left_inv = tf.squeeze(tf.matrix_inverse(K_left), 1)
    # K_right=tf.squeeze(K_right,1)
    R_left_trans = tf.transpose(tf.squeeze(R_left, axis=1), perm=[0, 2, 1])
    R_right_trans = tf.transpose(tf.squeeze(R_right, axis=1), perm=[0, 2, 1])
    c_left = -tf.matmul(R_left_trans, tf.squeeze(t_left, axis=1))
    c_right = -tf.matmul(R_right_trans, tf.squeeze(t_right, axis=1))  # (B, D, 3, 1)
    c_relative = tf.subtract(c_right, c_left)  # b,3,1
    P = tf.matmul(tf.squeeze(tf.matmul(K_right, R_right),1), tf.matmul(R_left_trans, K_left_inv))
    Q = tf.matmul(tf.squeeze(tf.matmul(K_right, R_right),1), c_relative)
    P = tf.matmul(P, pixel_grids)#b,3,[h*w]
    return P, Q
def grad_d(P,Q,x,axis=0):
    p0=tf.slice(P,[0,axis,0],[-1,1,-1])
    p2=tf.slice(P,[0,2,0],[-1,1,-1])
    q0=tf.slice(Q,[0,axis,0],[-1,1,-1])
    q2=tf.slice(Q,[0,2,0],[-1,1,-1])
    up=q2*x-q0
    div=p2*x-p0
    mask=tf.cast(div==0.0,dtype=tf.float32)
    div=mask*div+1e-7+(1-mask)*div
    depth=tf.divide(up,div)
    return depth
