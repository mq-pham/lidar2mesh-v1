import torch





def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point_adaptive(radius, nsample, xyz, new_xyz, max_radius_expansion=10.0):
    """
    Input:
        radius: initial local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, 3)
        new_xyz: query points, (B, M, 3)
        max_radius_expansion: maximum factor to expand the radius if no points are found
    Return:
        group_idx: grouped points index, (B, M, nsample)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # Initialize group_idx
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    current_radius = torch.full((B, S), radius, device=device)
    
    while True:
        mask = sqrdists <= current_radius.unsqueeze(-1) ** 2

        # Count number of neighbors for each query point
        neighbor_count = mask.sum(dim=-1)

        # For points with no neighbors, expand radius
        no_neighbors = neighbor_count == 0
        if not no_neighbors.any():
            break

        # Expand radius only for points with no neighbors
        expansion_mask = no_neighbors & (current_radius < radius * max_radius_expansion)
        current_radius[expansion_mask] *= 2

        if not expansion_mask.any():
            print("Warning: Some points still have no neighbors after maximum radius expansion")
            break
    
    # Assign N to points still without enough neighbors
    group_idx[~mask] = N-1
    
    # Sort and limit to nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # Handle cases where there are still not enough neighbors
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    final_mask = group_idx == N-1
    group_idx[final_mask] = group_first[final_mask]
    
    return group_idx, current_radius


def gather(points, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, S]
    Return:
        out: indexed points data, [B, C, S]
    """
    B, C, N = points.shape
    _, S = idx.shape
    
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)
    out = torch.gather(points, dim=2, index=idx_expanded)
    
    return out


def grouping(points, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, S, nsample]
    Return:
        out: grouped points data, [B, C, S, nsample]
    """
    B, C, N = points.shape
    _, S, nsample = idx.shape
    
    #idx_clamped = idx.clamp(0, N-1)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
    out = torch.gather(points.unsqueeze(3).expand(-1, -1, -1, nsample), dim=2, index=idx_expanded)
    
    return out