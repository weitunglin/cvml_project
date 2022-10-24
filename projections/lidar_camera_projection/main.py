import os

import matplotlib
import matplotlib.pyplot as plt
import open3d

from utils import *


def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

    # create open3d point cloud and axis
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(imgfov_pc_velo)
    entities_to_draw = [pcd, mesh_frame]

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue

        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Open3d boxes
        boxes3d_pts = open3d.utility.Vector3dVector(boxes3d_pts.T)
        box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        box.color = [1, 0, 0]
        entities_to_draw.append(box)

    # Draw
    open3d.visualization.draw_geometries([*entities_to_draw],
                                         front=[-0.9945, 0.03873, 0.0970],
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    with np.printoptions(threshold=100):
        print(pts_velo[:100])

    print("pts velo: ")
    print(pts_velo.shape)
    # apply projection
    pts_2d = project_to_image(pts_velo.T, proj_velo2cam2)
    # x, y = pts_2d

    # plt.subplot(2, 2, 1)
    # _ = plt.hist(x, bins="auto")
    # plt.subplot(2, 2, 2)
    # _ = plt.hist(y, bins="auto")

    # normalize pts_2d to image size
    # x = (x - np.mean(x)) / np.std(x) * 1280
    # y = (y - np.mean(y)) / np.std(y) * 720
    # pts_2d[0] = x
    # pts_2d[1] = y


    with np.printoptions(threshold=100):
        print(pts_2d[:100])

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
    print("=>depth")
    print(np.max(imgfov_pc_cam2[2]), np.min(imgfov_pc_cam2))
    print(imgfov_pc_cam2[2])
    pts_velo = pts_velo[inds, :]

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    z = (pts_velo[:, 2] - np.min(pts_velo[:, 2])) / np.ptp(pts_velo[:, 2]) * 255
    print(cmap)
    print(z)

    depth_map = np.zeros((img_height, img_width, 1))

    for i in range(imgfov_pc_pixel.shape[1]):
        depth_map[int(imgfov_pc_pixel[1, i]), int(imgfov_pc_pixel[0, i])] = (((100 - imgfov_pc_cam2[2, i]) / 100) * 255)
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        # color = (0, 255, 0)
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    

    plt.subplot(2, 1, 1)
    plt.imshow(depth_map, cmap="gray")
    plt.subplot(2, 1, 2)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img, depth_map


if __name__ == '__main__':
    # Load image, calibration file, label bbox
    matplotlib.use("TkAgg")

    timestamp = "1664426889016488790"

    rgb = cv2.cvtColor(cv2.imread(os.path.join(f'data/raw_{timestamp}.jpg')), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1920, 1080))
    cv2.imwrite(f"data/p_rgb_{timestamp}.png", rgb)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_calib_file('data/000114_calib.txt')

    # Load labels
    labels = load_label('data/000114_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan(f'data/lidar_{timestamp}.bin')[:, :3]
    # pc_velo = load_velo_scan(f'data/000114.bin')[:, :3]

    # np.savetxt("lidar_kitti.txt", load_velo_scan(f'data/000114.bin')[:, :3])
    # np.savetxt("lidar_itri.txt", load_velo_scan(f'data/lidar_{timestamp}.bin')[:, :3])

    # render_image_with_boxes(rgb, labels, calib)
    # render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    img, depth_map = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
    cv2.imshow("d", depth_map)
    print(depth_map)
    cv2.imwrite(f"data/p_dmap_{timestamp}.png", depth_map)
    cv2.imwrite(f"data/p_{timestamp}.png", img)
    # cv2.waitKey(0)
