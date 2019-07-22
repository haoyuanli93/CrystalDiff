"""
This module is used to create classes describing crystals

fs, um are the units
"""

import numpy as np

from CrystalDiff import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi


class CrystalBlock3D:
    def __init__(self):
        #############################
        # First level of parameters
        ##############################
        # This is just a default value
        bragg_energy = 6.95161 * 2  # kev

        # Reciprocal lattice in um^-1
        self.h = np.array((0, util.kev_to_wave_number(bragg_energy), 0.), dtype=np.float64)

        # The normal direction of the front surface of the crystal
        self.normal = np.array((0, -1., 0), dtype=np.float64)

        # The point that the surface through
        self.surface_point = np.array((0., 0., 0.), dtype=np.float64)

        # The thickness of the crystal in um
        self.d = 100.

        # zero component of electric susceptibility's fourier transform
        self.chi0 = complex(-0.15124e-4, 0.13222E-07)

        # h component of electric susceptibility's fourier transform
        self.chih_sigma = complex(0.37824E-05, -0.12060E-07)

        # hbar component of electric susceptibility's fourier transform
        self.chihbar_sigma = complex(0.37824E-05, -0.12060E-07)

        # h component of electric susceptibility's fourier transform
        self.chih_pi = complex(0.37824E-05, -0.12060E-07)

        # hbar component of electric susceptibility's fourier transform
        self.chihbar_pi = complex(0.37824E-05, -0.12060E-07)

        #############################
        # Second level of parameters. These parameters can be handy in the simulation
        #############################
        self.dot_hn = np.dot(self.h, self.normal)
        self.h_square = self.h[0] ** 2 + self.h[1] ** 2 + self.h[2] ** 2
        self.h_len = np.sqrt(self.h_square)

    def set_h(self, reciprocal_lattice):
        self.h = np.array(reciprocal_lattice)
        self._update_dot_nh()
        self._update_h_square()

    def set_surface_normal(self, normal):
        """
        Define the normal direction of the incident surface. Notice that, this algorithm assumes that
        the normal vector points towards the interior of the crystal.

        :param normal:
        :return:
        """
        self.normal = normal
        self._update_dot_nh()

    def set_surface_position(self, position):
        """

        :param position:
        :return:
        """
        self.surface_point = position

    def set_thickness(self, d):
        """
        Set the lattice thickness
        :param d:
        :return:
        """
        self.d = d

    def set_chi0(self, chi0):
        self.chi0 = chi0

    def set_chih_sigma(self, chih):
        self.chih_sigma = chih

    def set_chihbar_sigma(self, chihb):
        self.chihbar_sigma = chihb

    def set_chih_pi(self, chih):
        self.chih_pi = chih

    def set_chihbar_pi(self, chihb):
        self.chihbar_pi = chihb

    def _update_dot_nh(self):
        self.dot_hn = np.dot(self.normal, self.h)

    def _update_h_square(self):
        self.h_square = self.h[0] ** 2 + self.h[1] ** 2 + self.h[2] ** 2
        self.h_len = np.sqrt(self.h_square)

    def shift(self, displacement):
        self.surface_point += displacement

    def rotate(self, rot_mat):
        # change the position
        self.surface_point = np.ascontiguousarray(rot_mat.dot(self.surface_point))

        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.h = np.ascontiguousarray(rot_mat.dot(self.h))
        self.normal = np.ascontiguousarray(rot_mat.dot(self.normal))
