# This module include configuarions as well as global public variables
# Local public bariables are defined in each module
# Users can edit configurations to suit file system
# Don't edit golbal public variables here
import os

# Configurations
DATA_PATH = "/home/tai/UG4_Project/Data"
GALATEA_PATH = "/home/tai/UG4_Project/galatea"
ACTORS = ("Adam", "Beve", "Bonn", "Bria", "Dani", "Ella", "Esmo", "Haze", "Iren", "Jack", "Liam", "Paul", "Soph")
FORMATS = ("axa", "dof", "ea132", "qtn", "rov")
PLAY_SOFTWARE_PATH = "Hiroshi/MotionCapture/Export/gRigidBodies4"
PLAY_COMMAND = os.path.join(GALATEA_PATH, PLAY_SOFTWARE_PATH)

# Headers used to generate new trajectory file
ROV_HEADER = '''NO_OF_FRAMES	31734
FREQUENCY	100
DATA_INCLUDED	6DOF(rotation-vector)
ORDER	RVx RVy RVz Tx Ty Tz RMS
SOURCE_FILE	/afs/inf.ed.ac.uk/group/cstr/projects/galatea/DB/MotionCapture/Recordings_October_2014/Markers/Adam/CSTR/Adam_01_n.tsv,Head/Adam_01_n.dof.gz
RIGID_BODY_MARKERS	 1 2 3 4
PROGRAM_NAME	EstimateRigidBodyWithSegment,Conv2RotVecDOF
REFERENCE_FRAME	3
SEGMENT_HALF_WIDTH	2
SEGMENT_WINDOW_TYPE	rectangle
REFERENCE_MARKER_1	-0.372137 0.772435 0.235518
REFERENCE_MARKER_2	-0.328046 0.801676 0.0206238
REFERENCE_MARKER_3	-0.301764 0.805212 0.109519
REFERENCE_MARKER_4	-0.332634 0.816848 0.174299
REFERENCE_CENTRE	-0.333645 0.799043 0.13499
END_OF_HEADER
'''
AXA_HEADER = '''NO_OF_FRAMES	31734
FREQUENCY	100
DATA_INCLUDED	6DOF(axis-angle)
ORDER	Nx Ny Nz Angle Tx Ty Tz RMS
SOURCE_FILE	/afs/inf.ed.ac.uk/group/cstr/projects/galatea/DB/MotionCapture/Recordings_October_2014/Markers/Adam/CSTR/Adam_01_n.tsv,Head/Adam_01_n.dof.gz
RIGID_BODY_MARKERS	 1 2 3 4
PROGRAM_NAME	EstimateRigidBodyWithSegment,Conv2AxisAngleDOF
REFERENCE_FRAME	3
SEGMENT_HALF_WIDTH	2
SEGMENT_WINDOW_TYPE	rectangle
REFERENCE_MARKER_1	-0.372137 0.772435 0.235518
REFERENCE_MARKER_2	-0.328046 0.801676 0.0206238
REFERENCE_MARKER_3	-0.301764 0.805212 0.109519
REFERENCE_MARKER_4	-0.332634 0.816848 0.174299
REFERENCE_CENTRE	-0.333645 0.799043 0.13499
END_OF_HEADER
'''
DOF_HEADER = '''NO_OF_FRAMES	31734
FREQUENCY	100
DATA_INCLUDED	6DOF(row-major-order)
ORDER	R11 R12 R13 R21 R22 R23 R31 R32 R33 Tx Ty Tz RMS
SOURCE_FILE	/afs/inf.ed.ac.uk/group/cstr/projects/galatea/DB/MotionCapture/Recordings_October_2014/Markers/Adam/CSTR/Adam_01_n.tsv
RIGID_BODY_MARKERS	 1 2 3 4
PROGRAM_NAME	EstimateRigidBodyWithSegment
REFERENCE_FRAME	3
SEGMENT_HALF_WIDTH	2
SEGMENT_WINDOW_TYPE	rectangle
REFERENCE_MARKER_1	-0.372137 0.772435 0.235518
REFERENCE_MARKER_2	-0.328046 0.801676 0.0206238
REFERENCE_MARKER_3	-0.301764 0.805212 0.109519
REFERENCE_MARKER_4	-0.332634 0.816848 0.174299
REFERENCE_CENTRE	-0.333645 0.799043 0.13499
END_OF_HEADER
'''
EA132_HEADER = '''NO_OF_FRAMES	31734
FREQUENCY	100
DATA_INCLUDED	6DOF(Euler-angles:Y->Z->X:global:degrees)
ORDER	EAY EAZ EAX Tx Ty Tz RMS
SOURCE_FILE	/afs/inf.ed.ac.uk/group/cstr/projects/galatea/DB/MotionCapture/Recordings_October_2014/Markers/Adam/CSTR/Adam_01_n.tsv,Head/Adam_01_n.dof.gz
RIGID_BODY_MARKERS	 1 2 3 4
PROGRAM_NAME	EstimateRigidBodyWithSegment,Conv2EulerDOF
REFERENCE_FRAME	3
SEGMENT_HALF_WIDTH	2
SEGMENT_WINDOW_TYPE	rectangle
REFERENCE_MARKER_1	-0.372137 0.772435 0.235518
REFERENCE_MARKER_2	-0.328046 0.801676 0.0206238
REFERENCE_MARKER_3	-0.301764 0.805212 0.109519
REFERENCE_MARKER_4	-0.332634 0.816848 0.174299
REFERENCE_CENTRE	-0.333645 0.799043 0.13499
END_OF_HEADER
'''
QTN_HEADER = '''NO_OF_FRAMES	31734
FREQUENCY	100
DATA_INCLUDED	6DOF(Quaternion)
ORDER	Qw Qx Qy Qz Tx Ty Tz RMS
SOURCE_FILE	/afs/inf.ed.ac.uk/group/cstr/projects/galatea/DB/MotionCapture/Recordings_October_2014/Markers/Adam/CSTR/Adam_01_n.tsv,Head/Adam_01_n.dof.gz
RIGID_BODY_MARKERS	 1 2 3 4
PROGRAM_NAME	EstimateRigidBodyWithSegment,Conv2QuaternionDOF
REFERENCE_FRAME	3
SEGMENT_HALF_WIDTH	2
SEGMENT_WINDOW_TYPE	rectangle
REFERENCE_MARKER_1	-0.372137 0.772435 0.235518
REFERENCE_MARKER_2	-0.328046 0.801676 0.0206238
REFERENCE_MARKER_3	-0.301764 0.805212 0.109519
REFERENCE_MARKER_4	-0.332634 0.816848 0.174299
REFERENCE_CENTRE	-0.333645 0.799043 0.13499
END_OF_HEADER
'''
HEADERS = {
    "axa": AXA_HEADER,
    "dof": DOF_HEADER,
    "ea132": EA132_HEADER,
    "qtn": QTN_HEADER,
    "rov": ROV_HEADER
}

# Global public variables among the system
# Codes in other module will import config to use them
# These vairables are used as global variables when training system is running,
# Don't edit them
ALL_TRAJECTORY = None
STANDARDIZED = False
DATA_MEAN = None
DATA_STD = None
DATA_FORMAT = None
DATA_LABELS = None
PCA = None
TRIM_LENGTH = None
AVERAGE_SPECTRA = None
ALL_COMPLEX_DATA = None

def set_trim_length(length):
    # Setting the length of slicing data
    # This function should be only invoke once per training
    global TRIM_LENGTH

    if TRIM_LENGTH is not None:
        raise Exception("Trim length is already setted up!")

    TRIM_LENGTH = length
