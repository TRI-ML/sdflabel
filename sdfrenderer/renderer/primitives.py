import torch


def inside_circle(
        K,
        grid_2d,
        vertex_2d,
        vertex_3d,
        normals,
        diam=0.07,
        depth_constant=100,
        softclamp=True,
        softclamp_constant=3,
        add_bg=False
):
    """
    Compute output color probabilies per pixel using 2d circles as primitives.
    Compute distances between screen points and the projected 3d vertex points.
    Clamp distances over diam value to form circles.
    Use a rendering function to compute final color probabilities.

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        grid_2d (torch.Tensor): 2D pixel grid (1,N,2)
        vertex_2d (torch.Tensor): Locations of the object vertices on 2D screen (N,2)
        vertex_3d (torch.Tensor): Locations of the object vertices (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        diam (float): Diameter of the primitive
        depth_constant (float): Softmax depth constant
        softclamp (bool): Use Sigmoid if true, clamp values if false
        softclamp_constant (float): Multiplier if Sigmoid is used
        add_bg (float): Add background if True
    """
    # Define precision and device
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype

    # Vectors from vertices to grid points
    diff = vertex_2d[:, :2].view([-1, 1, 2]) - grid_2d.to(dtype)

    # Simple clamping vs differentiable sigmoid clamping
    if softclamp:
        dist_to_point = torch.sigmoid(
            ((abs(K[0, 0] * diam /
                  (vertex_3d[:, 2] + eps)).unsqueeze(-1)) - diff.pow(2).sum(-1).sqrt()) * softclamp_constant
        )
    else:
        dist_to_point = torch.clamp((abs(K[0, 0] * diam / (vertex_3d[:, 2] + eps)).unsqueeze(-1)) -
                                    diff.pow(2).sum(-1).sqrt(),
                                    min=0.)

    # Compute the depth-based accumulation function
    dist_to_point = (dist_to_point > 0).detach().to(K.dtype)
    z = -vertex_3d[:, 2:]  # - inverse depth
    z_norm = torch.norm(z, p=2, dim=0).detach()
    # z = z.div(z_norm.unsqueeze(0) + self.eps) * depth_constant  # compute norm and normalize
    z = torch.clamp((z.div(z_norm.unsqueeze(0) + eps) + 1),
                    min=0) * depth_constant  # compute norm and normalize

    # - add background
    if add_bg:
        z_bg = (z.min() - 1).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([z, z_bg])
        dist_to_point = torch.cat([dist_to_point, torch.ones_like(dist_to_point[:1, :])])

    # - resulting probability
    prob_z = torch.softmax(z * dist_to_point, dim=0) * dist_to_point
    prob_color = prob_z.unsqueeze(1).expand(-1, 3, dist_to_point.size(-1))  # extend to RGB

    return prob_color


def inside_circle_opt(
        K,
        grid_2d,
        vertex_2d,
        vertex_3d,
        normals,
        diam=0.06,
        depth_constant=10000,
        softclamp=True,
        softclamp_constant=5,
        add_bg=True
):
    """
    Compute output color probabilies per pixel using 2d circles as primitives.
    Use sparse matrices to store primitives and save memory.
    Compute distances between screen points and the projected 3d vertex points.
    Form circles using a sigmoid function, i.e. probabilistic distance.
    Use a rendering function to compute final color probabilities.

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        grid_2d (torch.Tensor): 2D pixel grid (1,N,2)
        vertex_2d (torch.Tensor): Locations of the object vertices on 2D screen (N,2)
        vertex_3d (torch.Tensor): Locations of the object vertices (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        diam (float): Diameter of the primitive
        depth_constant (float): Softmax depth constant
        softclamp (bool): Use Sigmoid if true, clamp values if false
        softclamp_constant (float): Multiplier if Sigmoid is used
        add_bg (float): Add background if True
    """
    # Define precision and device
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype
    x_px = (K[0, 2].int().item() * 2)
    y_px = (K[1, 2].int().item() * 2)

    # Compute primitives in sparse representation
    dist_primitive = grid_2d.to(dtype).pow(2).sum(-1).sqrt()
    diam_primitives = abs(K[0, 0] * diam / (vertex_3d[:, 2] + eps))

    # Simple clamping vs differentiable sigmoid clamping
    if softclamp:
        primitives = torch.sigmoid((diam_primitives.unsqueeze(-1) - dist_primitive) * softclamp_constant)
    else:
        primitives = torch.clamp(diam_primitives.unsqueeze(-1) - dist_primitive, min=0)
    ids_sparse_size = [vertex_2d.size(0), * grid_2d[0].size()]
    ids_sparse = (
            grid_2d.to(dtype).expand(ids_sparse_size) + vertex_2d.unsqueeze(-2).expand(ids_sparse_size)
    ).contiguous().long()

    # - clamp indices
    ids_sparse_l = torch.Tensor([[0, 0]]).to(device)  # lower bound
    ids_sparse_u = torch.Tensor([[x_px - 1, y_px - 1]]).to(device)  # upper bound
    ids_sparse = torch.max(torch.min(ids_sparse, ids_sparse_u.long()), ids_sparse_l.long())

    # - define primitives with positions and convert to the dense representation
    third_dim = torch.arange(diam_primitives.size()[0]
                             ).unsqueeze(-1).unsqueeze(-1).expand(ids_sparse[:, :, :1].size())
    ids_sparse = torch.cat([third_dim.to(device), ids_sparse], dim=2)
    sparse_prim_dist = torch.sparse.FloatTensor(
        ids_sparse[:, :, [0, 2, 1]].contiguous().view(-1, 3).t(),
        primitives.view(-1).float(), torch.Size([primitives.size(0), y_px, x_px])
    )  # sparce only implemented for float

    # Compute the depth-based accumulation function
    z = -vertex_3d[:, 2:]
    z_norm = torch.norm(z, p=2, dim=0).detach()
    z = torch.clamp((z.div(z_norm.unsqueeze(0) + eps) + 1),
                    min=0) * depth_constant  # compute norm and normalize

    # - add background
    if add_bg:
        z_bg = (z.min() - 1).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([z, z_bg])
        dist_to_point = torch.cat([
            sparse_prim_dist.to_dense().view(-1, y_px * x_px),
            torch.ones(1, y_px * x_px).to(sparse_prim_dist.device, dtype)
        ])
    else:
        dist_to_point = (sparse_prim_dist.to_dense().view(primitives.size(0), -1)).to(dtype)

    # - resulting probability
    dist_to_point = (dist_to_point > 0).detach().to(dtype)
    prob_z = torch.softmax(z.masked_fill((1 - dist_to_point).bool(), torch.finfo(dtype).min), dim=0) * dist_to_point
    prob_color = prob_z.unsqueeze(1).expand(-1, 3, dist_to_point.size(-1))  # extend to RGB

    return prob_color


def inside_surfel(
        K,
        grid_2d,
        vertex_2d,
        vertex_3d,
        normals,
        diam=0.03,
        depth_constant=150,
        softclamp=True,
        softclamp_constant=5,
        add_bg=True
):
    """
    Compute output color probabilies per pixel using 3d tangent disks as primitives.
    Use normals and 3d vertex points to compute planes's [x, y, z] coordinates.
    Compute distances between plane points and the actual 3d vertex points.
    Clamp distances over diam value to form tangent disks.
    Use a rendering function to compute final color probabilities

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        grid_2d (torch.Tensor): 2D pixel grid (1,N,2)
        vertex_2d (torch.Tensor): Locations of the object vertices on 2D screen (N,2)
        vertex_3d (torch.Tensor): Locations of the object vertices (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        diam (float): Diameter of the primitive
        depth_constant (float): Softmax depth constant
        softclamp (bool): Use Sigmoid if true, clamp values if false
        softclamp_constant (float): Multiplier if Sigmoid is used
        add_bg (float): Add background if True
    """
    # Define precision and device
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype

    # Find 3D points on a plane: z = normals.dot(vertex_3d) / normals.dot(K.inv() @ vertex_2d)
    n_v3d = torch.bmm(normals.unsqueeze((-2)), vertex_3d.unsqueeze(-1))
    Kinv_grid2d = (
            K.float().inverse().to(dtype)
            @ torch.cat([grid_2d[0].to(dtype),
                         torch.ones(grid_2d.shape[1], 1).to(device, dtype)],
                        dim=-1).t()
    ).t().unsqueeze(0)
    n_Kinv_grid2d = torch.bmm(Kinv_grid2d.expand(normals.size(0), Kinv_grid2d.size(1), -1), normals.unsqueeze(-1))
    n_Kinv_grid2d[n_Kinv_grid2d.abs() < 0.01] = torch.Tensor([eps]).to(device, dtype)
    z = n_v3d.expand_as(n_Kinv_grid2d) / n_Kinv_grid2d
    grid_3d = Kinv_grid2d * z

    # Compute vectors from vertices to grid points in 3d
    vectors_to_point = vertex_3d.view([-1, 1, 3]) - grid_3d

    if softclamp:
        dist_to_point = torch.sigmoid((diam - vectors_to_point.pow(2).sum(-1).sqrt()) * softclamp_constant)
    else:
        dist_to_point = torch.clamp(diam - vectors_to_point.pow(2).sum(-1).sqrt(), min=0)

    # Clean up
    del vectors_to_point, grid_3d

    # Compute the depth-based accumulation function
    dist_to_point = (dist_to_point > 0).detach().to(dtype)
    z = -z[:, :, 0] * dist_to_point
    z_norm = torch.norm(z, p=2, dim=0).detach()
    z = torch.clamp((z.div(z_norm.unsqueeze(0) + eps) + 1),
                    min=0) * depth_constant  # compute norm and normalize

    # - add background
    if add_bg:
        z2d = -vertex_3d[:, 2:] * depth_constant
        z_bg = (z2d.min() - 1).unsqueeze(-1).unsqueeze(-1).expand_as(dist_to_point[:1, :])
        z = torch.cat([z, z_bg])
        dist_to_point = torch.cat([dist_to_point, torch.ones_like(dist_to_point[:1, :])])

    # - resulting probability
    prob_z = torch.softmax(z.masked_fill((1 - dist_to_point).bool(), torch.finfo(dtype).min), dim=0) * dist_to_point
    prob_color = prob_z.unsqueeze(1).expand(-1, 3, prob_z.size(-1))  # extend to RGB

    return prob_color