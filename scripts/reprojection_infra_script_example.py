from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

# PROJECTION FUNCTIONS


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        points_3d = torch.matmul(T, points)
        cam_points = torch.matmul(K, points_3d)[:, :3, :]

        pix_coords = cam_points[:, :2, :] / cam_points[:, 2, :].unsqueeze(1)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)

        pix_coords_norm = pix_coords.permute(0, 2, 3, 1).contiguous()


        return pix_coords_norm


# PROJECTION PARAMETERS


def init_retroprojection(BATCH_SIZE = 1, WIDTH_INFRA = 382, HEIGHT_INFRA = 288, f_INFRA = 620, WIDTH_ZED = 1920, HEIGHT_ZED = 1080, f_ZED = 1000):
    
    PARAMS_INFRA = [f_INFRA, f_INFRA, WIDTH_INFRA / 2, HEIGHT_INFRA / 2]
    K_INFRA = [[PARAMS_INFRA[0], 0., PARAMS_INFRA[2], 0],
               [0., PARAMS_INFRA[1], PARAMS_INFRA[3], 0],
               [0., 0., 1., 0]]
    K_INFRA = torch.from_numpy(np.vstack((np.array(K_INFRA), [0, 0, 0, 1])).astype(np.float32)).unsqueeze(0)

    
    PARAMS_ZED = [f_ZED, f_ZED, WIDTH_ZED / 2, HEIGHT_ZED / 2]
    K_ZED = [[PARAMS_ZED[0], 0., PARAMS_ZED[2], 0],
             [0., PARAMS_ZED[1], PARAMS_ZED[3], 0],
             [0., 0., 1., 0]]
    K_ZED = np.vstack((np.array(K_ZED), [-10, 10, 0, 1])).astype(np.float32)
    invK_ZED = torch.from_numpy(np.linalg.inv(K_ZED)).unsqueeze(0)

    T = get_translation_matrix(torch.Tensor([0, 0, 0]).unsqueeze(0))

    backproject = BackprojectDepth(BATCH_SIZE, HEIGHT_ZED, WIDTH_ZED)
    project = Project3D(BATCH_SIZE, HEIGHT_ZED, WIDTH_ZED)
    return WIDTH_INFRA,HEIGHT_INFRA,K_INFRA,WIDTH_ZED,HEIGHT_ZED,invK_ZED,T,backproject,project

WIDTH_INFRA, HEIGHT_INFRA, K_INFRA, WIDTH_ZED, HEIGHT_ZED, invK_ZED, T, backproject, project = init_retroprojection(get_translation_matrix, BackprojectDepth, Project3D)


# CHANGE W/ PATH TOWARDS YOUR DEPTH IMAGES

depth_path = "/home/marwane/Desktop/research_u2is/datasets/small_dataset/depth/Anthony-Blr/t_1621535652_3.png"
depth = np.array(Image.open(depth_path))
# plt.imshow(depth)
depth = torch.from_numpy(depth).unsqueeze(0) + 1e-6
cam_points = backproject(depth, invK_ZED)
pix_coords_norm = project(cam_points, K_INFRA, T)


# CHANGE W/ PATH TOWARDS YOUR INFRA IMAGES

infra_path = "/home/marwane/Desktop/research_u2is/datasets/small_dataset/infra/Anthony-Blr/t_1621535652_3.png"
infra = np.array(Image.open(infra_path))
X = infra
# plt.imshow(X)
# infra = torch.from_numpy(infra).unsqueeze(0).unsqueeze(0) + 1e-6


# CHANGE W/ PATH TOWARDS YOUR SEG IMAGES

seg_path = "/home/marwane/Desktop/research_u2is/datasets/small_dataset/semantic/Anthony-Blr/Left/t_1621535652_3.png.png"
seg = np.array(Image.open(seg_path))
# plt.imshow(seg)
#


# CROPPING SEG IMAGES

seg_new = np.zeros((HEIGHT_INFRA, WIDTH_INFRA, 4))
for i in range(HEIGHT_ZED):
    for j in range(WIDTH_ZED):
        infra_i = round(pix_coords_norm[0][i][j][1].item())
        infra_j = round(pix_coords_norm[0][i][j][0].item())
        print(infra_j, infra_j)
        if 0 <= infra_i < HEIGHT_INFRA and 0 <= infra_j < WIDTH_INFRA:
            seg_new[infra_i][infra_j] = seg[i][j]


# DISPLAY CROP

f, axarr = plt.subplots(1,3)
axarr[0].imshow(seg)
axarr[1].imshow(seg_new.astype(int))
axarr[2].imshow(X)






