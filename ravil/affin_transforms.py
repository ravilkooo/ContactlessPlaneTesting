import numpy as np


class Affin_Transforms:
    def __init__(self, mark1, mark2, img_shape):
        self.capt2algn = np.zeros((2,3))
        (_h, _w) = (img_shape[0], img_shape[1])
        _mean_a = (mark1.rot_angle + mark2.rot_angle) / 2
        _mean_diam = (mark1.diam + mark2.diam) / 2
        _cos_a, _sin_a = np.cos(np.deg2rad(_mean_a)), np.sin(np.deg2rad(_mean_a))
        _dx, _dy = mark1.x_pos, mark1.y_pos
        self.capt2algn[0,:] = np.array([ _cos_a, -_sin_a, ( -_dx*_cos_a + (_dy)*_sin_a ) ])
        self.capt2algn[1,:] = np.array([ -_sin_a, -_cos_a, ( _dx*_sin_a + (_dy)*_cos_a ) ])

        self.capt2insc = self.capt2algn.copy()
        left_m_indent = abs(([[1920, 1080, 1], [0, 1080, 1], [1920, 0, 1], [0, 0, 1]] @ self.capt2algn.T)[:,0].min())
        self.capt2insc[:,2] += [left_m_indent, _mean_diam]

        mm2pix_ratio = 10. / _mean_diam
        self.capt2real = self.capt2algn * mm2pix_ratio
        

    def get_affin_capt2algn(self):
        # captured TO aligned along axises
        return self.capt2algn
    

    def get_affin_capt2insc(self):
        # captured TO aligned along axises
        return self.capt2insc


    def get_affin_capt2real(self):
        return self.capt2real


    def get_all_affin_tr(self):
        return np.array([self.capt2algn, self.capt2insc, self.capt2real])