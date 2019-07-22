import numpy as np
from CrystalDiff import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi
two_pi = 2 * pi


class GaussianPulse3D:
    def __init__(self):
        self.x0 = np.zeros(3, dtype=np.float64)
        self.k0 = np.zeros(3, dtype=np.float64)
        self.n = np.zeros(3, dtype=np.float64)

        self.omega0 = 20000.  # PHz

        # Basically, this mean that initially, we are in the frame where
        # different components of the pulse decouple. Then we rotate
        # back in to the lab frame.
        self.sigma_x = 0.1  # fs
        self.sigma_y = 33.  # fs
        self.sigma_z = 33.  # fs

        self.sigma_mat = np.diag(np.array([self.sigma_x ** 2,
                                           self.sigma_y ** 2,
                                           self.sigma_z ** 2], dtype=np.float64))

        # Intensity. Add a coefficient for overall intensity.
        self.scaling = 1.

        # Polarization
        self.polar = np.array([1., 0., 0.], dtype=np.complex128)


class SquarePulse3D:
    def __init__(self):
        self.k0 = np.zeros(3, dtype=np.float64)
        self.a_val = 1.
        self.b_val = 1.
        self.c_val = 1.
        # Intensity. Add a coefficient for overall intensity.
        self.scaling = 1.
        # Polarization
        self.polar = np.array([1., 0., 0.], dtype=np.complex128)


class MultiGaussianPulse:
    def __init__(self):
        # Number of gaussian components that consists this pulse
        self.num = int(1000)  # Assume that is pulse is composed of 1000 short gaussian pulse
        # This is range within which all the gaussian centers reside
        self.pulse_duration = 10.  # fs

        ################################
        # Properties of the gaussian pulses ensemble
        # Notice that because the following quantities can not be negative, therefore
        # we here use log-normal distributions. So the following quantities are specified
        # with respect to log-normal distribution
        ################################
        # Pulse central energy
        self.central_energy = 10.  # kev
        # The variation of the central energy for gaussian pulses
        self.energy_extension = 0.002  # kev

        # The mean gaussian pulse magnitude
        self.gaussian_mag_mean = 1.
        # The variance of the magnitude
        self.gaussian_mag_std = 1.

        self.mean_sigma_y = 10  # um
        self.std_sigma_y = 10  # The standard deviation of the sigma y for each gaussian pulse

        self.mean_sigma_t = 0.2  # The mean width of the gaussian pulses
        self.std_sigma_t = 0.1  # The std

        ###############################
        # Generate quantities for each individual gaussian pulses
        ###############################
        # Extension perpendicular to the propagation direction
        self.sigma_y = util.get_lognormal(mean=self.mean_sigma_y,
                                          sigma=self.std_sigma_y,
                                          size=self.num)  # um
        # The time duration of this pulse
        self.sigma_t = util.get_lognormal(mean=self.mean_sigma_t,
                                          sigma=self.std_sigma_t,
                                          size=self.num)
        # The center of the wave packet
        self.x0 = np.random.uniform(low=0,
                                    high=self.pulse_duration,
                                    size=self.num)  # um

        # The initial phase of the central frequency
        self.phi0 = np.random.uniform(low=0,
                                      high=2 * pi,
                                      size=self.num)  # radian

        # The central energy
        self.energy0 = np.random.uniform(low=self.central_energy - self.energy_extension / 2.,
                                         high=self.central_energy + self.energy_extension / 2.,
                                         size=self.num)  # kev
        # The central angular frequency
        self.omega0 = util.kev_to_petahertz_angular_frequency(energy=self.energy0)  # petahertz

        # The overall magnitude
        self.magnitude = util.get_lognormal(mean=self.gaussian_mag_mean,
                                            sigma=self.gaussian_mag_std,
                                            size=self.num)

        ########################################################################

    # Defining all sorts of setting functions
    ########################################################################
    def set_mean_sigma_y(self, mean_sigma_y):
        self.mean_sigma_y = mean_sigma_y

    def set_std_sigma_y(self, std_sigma_y):
        self.std_sigma_y = std_sigma_y

    def set_mean_sigma_t(self, mean_sigma_t):
        self.mean_sigma_t = mean_sigma_t

    def set_std_sigma_t(self, std_sigma_t):
        self.std_sigma_t = std_sigma_t

    def set_mean_gaussian_mag(self, mean_gaussian_mag):
        self.gaussian_mag_mean = mean_gaussian_mag

    def set_std_gaussian_mag(self, std_gaussian_mag):
        self.gaussian_mag_std = std_gaussian_mag

    def set_central_energy(self, central_energy):
        self.central_energy = central_energy

    def set_energy_extension(self, energy_extension):
        self.energy_extension = energy_extension

    def set_gaussian_num(self, num):
        self.num = num

    def set_pulse_duration(self, duration):
        self.pulse_duration = duration

    def update_pulse_profile(self):
        # Extension perpendicular to the propagation direction
        self.sigma_y = util.get_lognormal(mean=self.mean_sigma_y,
                                          sigma=self.std_sigma_y,
                                          size=self.num)  # um
        # The time duration of this pulse
        self.sigma_t = util.get_lognormal(mean=self.mean_sigma_t,
                                          sigma=self.std_sigma_t,
                                          size=self.num)
        # The center of the wave packet
        self.x0 = np.random.uniform(low=0,
                                    high=self.pulse_duration,
                                    size=self.num)  # um

        # The initial phase of the central frequency
        self.phi0 = np.random.uniform(low=0,
                                      high=2 * pi,
                                      size=self.num)  # radian

        # The central energy
        self.energy0 = np.random.uniform(low=self.central_energy - self.energy_extension / 2.,
                                         high=self.central_energy + self.energy_extension / 2.,
                                         size=self.num)  # kev
        # The central angular frequency
        self.omega0 = util.kev_to_petahertz_angular_frequency(energy=self.energy0)  # petahertz

        # The overall magnitude
        self.magnitude = util.get_lognormal(mean=self.gaussian_mag_mean,
                                            sigma=self.gaussian_mag_std,
                                            size=self.num)

        print("Finish updating pulse profile.")

    ########################################################################
    # Functions to retrieve electric field values
    ########################################################################
    def get_field_txy(self, local_time, position_x, position_y):
        """

        :param local_time:
        :param position_x:
        :param position_y:
        :return:
        """
        # This means that there is only one gaussian sub-pulse in this pulse
        if self.num == int(1):

            # The phase holder
            phase = (position_x - self.x0) / c - local_time
            total_field = self.magnitude * np.exp(-np.square(phase) / 2. / self.sigma_t ** 2
                                                  - np.square(position_y) / 2. / self.sigma_y ** 2)
            total_field = total_field.astype(np.complex128)
            total_field *= (np.cos(self.omega0 * phase + self.phi0) +
                            1.j * np.sin(self.omega0 * phase + self.phi0))

        else:
            # When we have more than one pulse, first initialize some variables
            phase = (position_x - self.x0[0]) / c - local_time
            total_field = self.magnitude[0] * np.exp(-np.square(phase) / 2. / self.sigma_t[0] ** 2
                                                     - np.square(position_y) / 2. / self.sigma_y[0] ** 2)
            total_field = total_field.astype(np.complex128)
            total_field *= (np.cos(self.omega0[0] * phase + self.phi0[0]) +
                            1.j * np.sin(self.omega0[0] * phase + self.phi0[0]))

            # Loop through the rest of the pulses.
            for idx in range(1, self.num):
                # Calculate contribution this specific gaussian component
                phase = (position_x - self.x0[idx]) / c - local_time
                tmp = self.magnitude[idx] * np.exp(-np.square(phase) / 2. / self.sigma_t[idx] ** 2
                                                   - np.square(position_y) / 2. / self.sigma_y[idx] ** 2)
                tmp = tmp.astype(np.complex128)
                tmp *= (np.cos(self.omega0[idx] * phase + self.phi0[idx]) +
                        1.j * np.sin(self.omega0[idx] * phase + self.phi0[idx]))

                # Add this contribution to the total field holder
                total_field += tmp

        return total_field

    def get_field_wxy(self, angular_frequency, position_x, position_y):
        """

        :param angular_frequency:
        :param position_x:
        :param position_y:
        :return:
        """
        # This means that there is only one gaussian sub-pulse in this pulse
        if self.num == int(1):
            phase_holder = angular_frequency * (position_x - self.x0) / c + self.phi0
            total_field = self.magnitude * self.sigma_t * (np.cos(phase_holder) +
                                                           1.j * np.sin(phase_holder))
            total_field *= np.exp(-np.square(position_y) / 2. / self.sigma_y ** 2
                                  - np.square(angular_frequency - self.omega0) * self.sigma_t ** 2)
        else:
            # When we have more than one pulse, first initialize some variables
            phase_holder = angular_frequency * (position_x - self.x0[0]) / c + self.phi0[0]
            total_field = self.magnitude[0] * self.sigma_t[0] * (np.cos(phase_holder) +
                                                                 1.j * np.sin(phase_holder))
            total_field *= np.exp(-np.square(position_y) / 2. / self.sigma_y[0] ** 2
                                  - np.square(angular_frequency - self.omega0[0]) * self.sigma_t[0] ** 2)

            # Loop through the rest of the pulses.
            for idx in range(1, self.num):
                phase_holder = angular_frequency * (position_x - self.x0[idx]) / c + self.phi0[idx]
                tmp = self.magnitude[idx] * self.sigma_t[idx] * (np.cos(phase_holder) +
                                                                 1.j * np.sin(phase_holder))
                tmp *= np.exp(-np.square(position_y) / 2. / self.sigma_y[idx] ** 2
                              - np.square(angular_frequency - self.omega0[idx]) * self.sigma_t[idx] ** 2)

                total_field += tmp

        return total_field

    def get_field_tpy(self, local_time, wavenumber_x, position_y):
        """

        :param local_time:
        :param wavenumber_x:
        :param position_y:
        :return:
        """
        # This means that there is only one gaussian sub-pulse in this pulse
        if self.num == int(1):
            phase_holder = (c * local_time + self.x0) * wavenumber_x + self.phi0
            total_field = self.magnitude * self.sigma_t * c * (np.cos(phase_holder) +
                                                               1.j * np.sin(phase_holder))
            total_field *= np.exp(-np.square(position_y) / 2. / self.sigma_y ** 2
                                  - np.square(wavenumber_x * c - self.omega0) * self.sigma_t ** 2)

        else:
            # When we have more than one pulse, first initialize some variables
            phase_holder = (c * local_time + self.x0[0]) * wavenumber_x + self.phi0[0]
            total_field = self.magnitude[0] * self.sigma_t[0] * c * (np.cos(phase_holder) +
                                                                     1.j * np.sin(phase_holder))
            total_field *= np.exp(-np.square(position_y) / 2. / self.sigma_y[0] ** 2
                                  - np.square(wavenumber_x * c - self.omega0[0]) * self.sigma_t[0] ** 2)

            # Loop through the rest of the pulses.
            for idx in range(1, self.num):
                phase_holder = (c * local_time + self.x0[idx]) * wavenumber_x + self.phi0[idx]
                tmp = self.magnitude[idx] * self.sigma_t[idx] * c * (np.cos(phase_holder) +
                                                                     1.j * np.sin(phase_holder))
                tmp *= np.exp(-np.square(position_y) / 2. / self.sigma_y[idx] ** 2
                              - np.square(wavenumber_x * c - self.omega0[idx]) * self.sigma_t[idx] ** 2)
                total_field += tmp

        return total_field

    def get_beam_info(self):
        print("Pulse duration: {:.2f} fs".format(self.pulse_duration))
        print("Pulse width: {:.2f} um".format(self.mean_sigma_y))
        print("Pulse central energy: {:.2f} keV".format(self.central_energy))
        print("Gaussian sub pulse duration: {:.2f} fs".format(self.mean_sigma_t))
        print("Gaussian sub pulse magnitude: {:.2f} ".format(self.gaussian_mag_mean))
