# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry


from model.smpl import SMPL, JOINTSTYPE_ROOT, SMPLX
# from .get_model import JOINTSTYPES
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices", "smplx"]


class Rotation2xyz:
    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])   # to (*,3,3)    
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
        
        
        # import pdb; pdb.set_trace()
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz



class Rotation2vertices:    
# modify this to smplx version
    def __init__(self, device):
        self.device = device
        self.smplx_model = SMPLX().eval().to(device)  # change to smplx

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, **kwargs):                            #x==== torch.Size([20, 54, 6, 60])
        if pose_rep == "xyz":  # xyz no need vertices
            return x    
        # import pdb; pdb.set_trace()
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")
        
        # translation =False  # modify 
        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]  #torch.Size([20, 60, 23, 6])
        else:
            x_rotations = x
        
       
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask]) # torch.Size([1200, 23, 3, 3])     
        elif pose_rep == "smplx" or "smplx6d":     # smplx is (., 3,3)  current not
            rotations= geometry.rotation_6d_to_matrix(x_rotations[mask])  # to (*,3,3)       x_rotations -> [20, 60, 53, 6]
        else:
            raise NotImplementedError("No geometry for this one.")

       
        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0] #torch.Size([120, 3, 3])
            rotations = rotations[:, 1:]   #torch.Size([1200, 22, 3, 3]  // torch.Size([120, 52, 3, 3])


        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smplx_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta     #torch.Size([120, 10])

       
        # import pdb; pdb.set_trace()

        root_pose=global_orient.unsqueeze(1) #torch.Size([1200, 1, 3, 3])
        body_pose=rotations[:,:21]   #torch.Size([1200, 21, 3, 3])
        lhand_pose=rotations[:,21:36]  #torch.Size([1200, 15, 3, 3])
        rhand_pose=rotations[:,36:51]  #torch.Size([1200, 15, 3, 3])
        jaw_pose=rotations[:,51].unsqueeze(1) #torch.Size([1200, 1, 3, 3])
    
        out = self.smplx_model(betas=betas, global_orient=root_pose, body_pose=body_pose,left_hand_pose=lhand_pose,right_hand_pose=rhand_pose,jaw_pose=jaw_pose)  
       
       
        # joints = out[jointstype]
        joints_vertices= out["vertices"]  #torch.Size([120, 10475, 3])
        joints_3d= out["joints"]    #torch.Size([1200, 127, 3])
        # this place makes the jointstype='smplx' useless

        # set a flag to get joints or vertices

        vertices_only= False
        if vertices_only:
            # keep the original code part
            x_xyz = torch.empty(nsamples, time, joints_vertices.shape[1], 3, device=x.device, dtype=x.dtype)  #(20,60, 10475,3)
            x_xyz[~mask] = 0
            x_xyz[mask] = joints_vertices
            x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

            # the first translation root at the origin on the prediction
            if jointstype != "vertices":
                rootindex = JOINTSTYPE_ROOT[jointstype]
                x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

            if translation and vertstrans:  #true and false
                # the first translation root at the origin
                x_translations = x_translations - x_translations[:, :, [0]]

                # add the translation to all the joints
                x_xyz = x_xyz + x_translations[:, None, :, :]

            return x_xyz             #torch.Size([2, 10475, 3, 60])

        else: 
            # modify to the version of two outputs
            x_mesh = torch.empty(nsamples, time, joints_vertices.shape[1], 3, device=x.device, dtype=x.dtype)  #(20,60, 10475,3)
            x_mesh[~mask] = 0
            x_mesh[mask] = joints_vertices
            x_mesh = x_mesh.permute(0, 2, 3, 1).contiguous()  #torch.Size([20, 10475, 3, 60])
            
            x_joint3d = torch.empty(nsamples, time, joints_3d.shape[1], 3, device=x.device, dtype=x.dtype)  #(20,60, 127,3)
            x_joint3d[~mask] = 0
            x_joint3d[mask] = joints_3d
            x_joint3d = x_joint3d.permute(0, 2, 3, 1).contiguous() # torch.Size([20, 127, 3, 60])

            if translation and vertstrans:  #true and false
                # the first translation root at the origin
                x_translations = x_translations - x_translations[:, :, [0]]

                # add the translation to all the joints
                x_mesh = x_mesh + x_translations[:, None, :, :]
                x_joint3d= x_joint3d + x_translations[:, None, :, :]

            # return x_mesh, x_joint3d
            return x_joint3d