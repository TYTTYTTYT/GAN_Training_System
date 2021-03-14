import os
from configurations import ACTORS
from configurations import DATA_PATH
from configurations import GALATEA_PATH

# ACTORS = ["Adam", "Beve", "Bonn", "Bria", "Dani", "Ella", "Esmo", "Haze", "Iren", "Jack", "Liam", "Paul", "Soph"]

if __name__ == "__main__":
    for n in ACTORS:
        os.system('mkdir {d}/Trajectory_Data/Raw/{a}'.format(a=n, d=DATA_PATH))
        os.system('mkdir {d}/Trajectory_Data/Normalised/{a}'.format(a=n, d=DATA_PATH))

        stream = os.popen('ls {g}/d02/Recordings_October_2014/DOF-hiroshi/{a}/WS2/Head | grep {a}_.._.*[^z]$'.format(a=n, g=GALATEA_PATH))
        filenames = stream.read().split()
        for f in filenames:
            os.system('ln -s {g}/d02/Recordings_October_2014/DOF-hiroshi/{c}/WS2/Head/{a} {d}/Trajectory_Data/Raw/{b}'.format(a=f, b=n, c=n, g=GALATEA_PATH, d=DATA_PATH))

    for n in ACTORS:
        stream = os.popen('ls {g}/d02/Recordings_October_2014/DOF-hiroshi/{a}/WS2/Head/Normalised | grep {a}_.._.*[^z]$'.format(a=n, g=GALATEA_PATH))
        filenames = stream.read().split()
        for f in filenames:
            os.system('ln -s {g}/d02/Recordings_October_2014/DOF-hiroshi/{c}/WS2/Head/{a} {d}/Trajectory_Data/Normalised/{b}'.format(a=f, b=n, c=n, d=DATA_PATH, g=GALATEA_PATH))
