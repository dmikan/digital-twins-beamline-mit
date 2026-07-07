import numpy as np


class BeamlineTransform:

    H2 = np.array([
        [1,1],
        [1,-1]
    ],dtype=float)/np.sqrt(2)

    H4 = np.array([
        [1,1,1,1],
        [1,-1,1,-1],
        [1,1,-1,-1],
        [1,-1,-1,1]
    ],dtype=float)/2


    def encode(self, voltages):
        """
        voltages

        [V3,V6,V9,V10,V11,V12,V15,V18]

        ->
        physical parameters
        """

        voltages=np.asarray(voltages,dtype=float)

        L=voltages[:2]
        Q=voltages[2:6]
        P=voltages[6:]

        theta=np.concatenate([
            self.H2@L,
            self.H4@Q,
            self.H2@P
        ])

        return theta


    def decode(self, theta):
        """
        physical parameters

        ->
        voltages
        """

        theta=np.asarray(theta,dtype=float)

        L=theta[:2]
        Q=theta[2:6]
        P=theta[6:]

        voltages=np.concatenate([
            self.H2@L,
            self.H4@Q,
            self.H2@P
        ])

        return voltages