import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    NDCMultinomialRaysampler,
    look_at_view_transform
)
from pytorch3d.structures import Pointclouds


class LiDAR(torch.nn.Module):
    def __init__(
        self,
        elev,
        dist,
        image_size,
        min_depth,
        max_depth,
        znear=0.1,
        subsample=None,
        device=None,
        return_normals=True
    ):
        super().__init__()
        self.elev = elev
        self.dist = dist
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.znear = znear
        self.subsample = subsample
        self.device = device
        self.return_normals = return_normals
            
        # Pre-compute mesh rasterization settings
        self.mesh_raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            perspective_correct=False,
            clip_barycentric_coords=None,
            cull_backfaces=False,
            bin_size=None,
            max_faces_per_bin=None,
            cull_to_frustum=False,
        )
        
        # Pre-compute raysampler
        self.raysampler = NDCMultinomialRaysampler(
            image_height=self.image_size[0],
            image_width=self.image_size[1],
            n_pts_per_ray=1,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )

    def forward(self, meshes, azimuths):
        
        batch_size = meshes.verts_padded().shape[0]
        
        first_intersection_points_list = []

        for i in range(batch_size):
            R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=azimuths[i].unsqueeze(0), degrees=True, device=self.device)
            
            cameras = FoVPerspectiveCameras(
                fov=20.0,
                R=R,
                T=T,
                device=self.device
            )

            # Create rasterizers
            mesh_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.mesh_raster_settings)
            
            # Rasterize the mesh
            fragments = mesh_rasterizer(meshes[i])

            # Generate ray bundles
            ray_bundle = self.raysampler(cameras=cameras)
            ray_origins = ray_bundle.origins
            ray_directions = ray_bundle.directions

            # Create a mask to identify valid pixel locations
            mask = (fragments.pix_to_face[..., 0] >= 0)

            # Compute the 3D coordinates and features of the first intersection points
            mask_i = mask[0]
            ray_origins_i = ray_origins[0]
            ray_directions_i = ray_directions[0]
            zbuf_i = fragments.zbuf[0]            

            first_intersection_points_i = ray_origins_i[mask_i] + ray_directions_i[mask_i] * zbuf_i[mask_i]
            first_intersection_points_list.append(first_intersection_points_i)

        # Create a Pointclouds object
        point_cloud = Pointclouds(points=first_intersection_points_list, features=None)
        
        # Downsample the point cloud if needed
        if self.subsample is not None and self.subsample > 0:
            point_cloud = point_cloud.subsample(max_points=self.subsample)
            
        # Compute normals if needed
        if self.return_normals:
            point_cloud.estimate_normals(neighborhood_size=50, disambiguate_directions=True, assign_to_self=True)

        return point_cloud