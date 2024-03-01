# WF 6x6 MAMS 2000 pinned weak axis BUCKLE
from part import *
from material import *
from section import *
from optimization import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
# do not forget to start with this before doung anything on the GUI
session.journalOptions.setValues(recoverGeometry=COORDINATE)
# Add these statements, they are not written to the .rec file
from abaqus import *            # loaded by GUI but not on .rec
from abaqusConstants import *   # loaded by GUI but not on .rec
from caeModules import *        # maybe needed but not on .rec
# Need these to do calculations in Python 
import math                     #      
import numpy as np
import odbAccess
from abaqus import backwardCompatibility    # datum(s) to be deprecated !
backwardCompatibility.setValues(reportDeprecated=False)

""""
Open BUCKLE.txt, a file that contains information that will change the solution provided
"""
params = np.loadtxt('BUCKLE.txt')
lambd  = float(params[0])
nummode = int(params[1])
num = int(params[2])
jobname = 'WF6X6_'+str(num)

fo = open(jobname+"_buckle.txt", "w")
CF = 1.0 # applied CF (== 1.0 for BUCKLE, == BUCKLE load 1st mode from BUCKLE run)
Lcr = 2280. # column length [mm], lambda=1.0, Pcr=169752. Interacting. With H44, H45, H55.
L = lambd*Lcr
nModes = nummode

L2 = L/2. # model length, 908 in CAE
b = 152. # flange width, 146 in CAE
h = 152. # section height, 146 in CAE
tf = 6.35 # flange thickness, 0.0 in CAE
tw = 7.14 # web thickness, 0.0 in CAE
hf = h - tf # mid- to mid-surface height of web, not used in CAE
h2 = hf/2.
b2 = b/2.
# Create Model and call it "m"
Mdb() # delete any previouos mdb
mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
m = mdb.models['Model-1']
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-b2, -h2), 
    point2=(b2, -h2))
# Top flange
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(-b2, h2), 
    point2=(b2, h2))
# Web
mdb.models['Model-1'].sketches['__profile__'].Line(point1=(0.0, h2), 
    point2=(0.0, -h2))
# Sketch to Part
mdb.models['Model-1'].Part(dimensionality=THREE_D, name='Part-1', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].parts['Part-1'].BaseShellExtrude(depth=L2, sketch=
    mdb.models['Model-1'].sketches['__profile__'])
del mdb.models['Model-1'].sketches['__profile__']
# Call it "p" and visualize it
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
# Module: Property (first, insert visualization commands for this Module)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)

mdb.models['Model-1'].GeneralStiffnessSection(applyThermalStress=0, name=
    'Flange', poissonDefinition=DEFAULT, referenceTemperature=None, 
    stiffnessMatrix=(163370, 31996, 87165, 0, 0, 25649, 0, 0, 0, 489226, 0, 0, 0, 116521, 308006, 0, 0, 0, 0, 0, 91080), useDensity=OFF)
mdb.models['Model-1'].sections['Flange'].TransverseShearShell(k11=15788.0, k12=
    0.0, k22=15338.0)
mdb.models['Model-1'].GeneralStiffnessSection(applyThermalStress=0, name='Web', 
    poissonDefinition=DEFAULT, referenceTemperature=None, stiffnessMatrix=(
    158176, 32038, 88103, 0, 0, 26132, 0, 0, 0, 767573, 0, 0, 0, 152832, 420500, 0, 0, 0, 0, 0, 124985), useDensity=OFF)
mdb.models['Model-1'].sections['Web'].TransverseShearShell(k11=16378.0, k12=
    0.0, k22=15955.0)

mdb.models['Model-1'].parts['Part-1'].DatumCsysByTwoLines(CARTESIAN, line1= mdb.models['Model-1'].parts['Part-1'].edges.findAt((0.0, -(h-tf)/2., L2/4), ), 
    line2=mdb.models['Model-1'].parts['Part-1'].edges.findAt((0.0, -(h-tf)/4., 
    L2), ), name='Web')

mdb.models['Model-1'].parts['Part-1'].DatumCsysByTwoLines(CARTESIAN, line1=
    mdb.models['Model-1'].parts['Part-1'].edges.findAt((0.0, -(h-tf)/2., L2/4), ), 
    line2=mdb.models['Model-1'].parts['Part-1'].edges.findAt((3*b2/4, -(h-tf)/2., 
    L2), ), name='Flange')

mdb.models['Model-1'].parts['Part-1'].MaterialOrientation(
    additionalRotationField='', additionalRotationType=ROTATION_ANGLE, angle=
    180.0, axis=AXIS_3, fieldName='', localCsys=
    mdb.models['Model-1'].parts['Part-1'].datums[2], orientationType=SYSTEM, 
    region=Region(faces=mdb.models['Model-1'].parts['Part-1'].faces.findAt(((
    0.0, -(h - tf)/6, L2/2), (-1.0, 0.0, 0.0)), )))
# Parameterized version

# Flange
mdb.models['Model-1'].parts['Part-1'].MaterialOrientation(
    additionalRotationField='', additionalRotationType=ROTATION_ANGLE, angle=
    0.0, axis=AXIS_3, fieldName='', localCsys=
    mdb.models['Model-1'].parts['Part-1'].datums[3], orientationType=SYSTEM, 
    region=Region(faces=mdb.models['Model-1'].parts['Part-1'].faces.findAt(
    ((-b2/2, -(h - tf)/2, L/3), (0.0, -1.0, 0.0)), ((b2/2, -(h - tf)/2, L/6), (0.0, -1.0, 0.0)), 
    ((-b2/2, (h - tf)/2, L/3), (0.0, -1.0, 0.0)), ((b2/2, (h - tf)/2, L/6), (0.0, -1.0, 0.0)), )))


mdb.models['Model-1'].parts['Part-1'].Set(faces=
    mdb.models['Model-1'].parts['Part-1'].faces.findAt(
    ((0.0, -(h - tf)/6, L2/2), (-1.0, 0.0, 0.0)), ), name='Set-2')
    
mdb.models['Model-1'].parts['Part-1'].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=
    mdb.models['Model-1'].parts['Part-1'].sets['Set-2'], sectionName='Web', 
    thicknessAssignment=FROM_SECTION)
   
mdb.models['Model-1'].parts['Part-1'].Set(faces=
    mdb.models['Model-1'].parts['Part-1'].faces.findAt(
   ((-b2/2, -(h - tf)/2, L/3), (0.0, -1.0, 0.0)), ((b2/2, -(h - tf)/2, L/6), (0.0, -1.0, 0.0)), 
    ((-b2/2, (h - tf)/2, L/3), (0.0, -1.0, 0.0)), ((b2/2, (h - tf)/2, L/6), (0.0, -1.0, 0.0)),), name='Set-3')    

mdb.models['Model-1'].parts['Part-1'].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=
    mdb.models['Model-1'].parts['Part-1'].sets['Set-3'], sectionName='Flange', 
    thicknessAssignment=FROM_SECTION)
    
# Assembly, Independent, from Part
r = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=r)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Model-1'].rootAssembly.Instance(dependent=OFF, name='Part-1-1', 
    part=mdb.models['Model-1'].parts['Part-1'])
# Define
i = r.Instance # >>> STEP follows <<<
# RP defined here to use it in RICKS termination control
mdb.models['Model-1'].rootAssembly.ReferencePoint(point=(0.0, 0.0, 0.0))
RP4 = r.referencePoints.keys()[-1] # assign newest RP to RP4 variable to use below
mdb.models['Model-1'].rootAssembly.Set(name='Set-RP', referencePoints=(
    mdb.models['Model-1'].rootAssembly.referencePoints[RP4], ))
# Module: Step (first, insert visualization commands for this Module)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
# ===========================================================================
# >>> Step-1 for BUCKLE analysis. Comment for RIKS. Uncomment for BUCKLE. 
mdb.models['Model-1'].BuckleStep(name='Step-1', numEigen=nModes, 
    vectors=2*nModes, maxIterations=100, previous='Initial')
#mdb.models['Model-1'].BuckleStep(name='Step-1', maxEigen=10.0, numEigen=10, 
#    vectors=30, previous='Initial')
# >>> Step-2 for RIKS. Comment for BUCKLE. Uncomment for RIKS
#mdb.models['Model-1'].StaticRiksStep(name='Step-2', nlgeom=ON, previous=
#    'Initial',dof=3, maximumDisplacement=tf, nodeOn=ON, region=
#    mdb.models['Model-1'].rootAssembly.sets['Set-RP'])# U3_max<=tf
#
# lambda : load factor. Delta_lambda : increment load factor of curent load_incr
# Set initialArcInc = 0.1 to get a result at 10% of CF
# Set maxArcInc=initialArcInc get results on equally spaced Arc increments. Default=1e+36
# Set minArcInc = 0.0 => defaults to min(initialArcInc,1E-5*totalArcLength)
# Set totalArcLength = 1.0 so that 
#  initialArcInc*totalArcLength => Delta_lambda_ini (load on 1st load_incr)
#  CF_ini = initialArcInc*totalArcLength*CF, whith CF = bifurcation load (input)
# Set maxNumInc=120 to get 120 frames, @ 1 fps->2 min video
#mdb.models['Model-1'].steps['Step-2'].setValues(initialArcInc=0.1, 
#    maxArcInc=0.1, minArcInc=0.0, totalArcLength=1.0, maxNumInc=120)
# ===========================================================================
# Interaction, rigid body to apply the load
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)

mdb.models['Model-1'].rootAssembly.Set(edges=
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.findAt(
    ((0.0, (h - tf)/4, 0.0),), ((b2/4, (h - tf)/2, 0.0),), ((-(3*b2/4), (h - tf)/2, 0.0),), 
    ((b2/4, -(h - tf)/2, 0.0),), ), name='t_Set-1')
        
mdb.models['Model-1'].RigidBody(name='Constraint-1', refPointRegion=Region(
    referencePoints=(mdb.models['Model-1'].rootAssembly.referencePoints[RP4], )), 
    tieRegion=mdb.models['Model-1'].rootAssembly.sets['t_Set-1'])
# Module Load (same visualization as Interaction)
# Create Set-3 to name the RP
mdb.models['Model-1'].rootAssembly.Set(name='Set-3', referencePoints=(
    mdb.models['Model-1'].rootAssembly.referencePoints[RP4], ))
# >>> For BUCKLE, apply a reference load CF. Comment for RIKS
mdb.models['Model-1'].ConcentratedForce(cf3=CF, createStepName='Step-1', 
    distributionType=UNIFORM, field='', localCsys=None, name='Load-1', region=
    mdb.models['Model-1'].rootAssembly.sets['Set-3'])#CF=1.0 is compression
# >>> For RIKS, on Step-2, apply the buckling load found with BUCKLE Step-1
# Comment for BUCKLE. Uncomment for RIKS
#mdb.models['Model-1'].ConcentratedForce(cf3=CF, createStepName='Step-2', 
#    distributionType=UNIFORM, field='', localCsys=None, name='Load-1', region=
#    mdb.models['Model-1'].rootAssembly.sets['Set-3'])# CF
#
# Parameterized version
mdb.models['Model-1'].rootAssembly.Set(edges=
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].edges.findAt(
    (((3*b2/4), -(h - tf)/2, L2),),(((3*b2/4), (h - tf)/2, L2),), ((0.0, -(h - tf)/4, L2),), 
    ((-b2/4, -(h - tf)/2, L2),), ((-b2/4, (h - tf)/2, L2),), ), name='Set-4')
# Apply ZSYMM : U3=UR1=UR2=0
mdb.models['Model-1'].ZsymmBC(createStepName='Initial', localCsys=None, name=
    'BC-1', region=mdb.models['Model-1'].rootAssembly.sets['Set-4'])
# Constrain RBM 
mdb.models['Model-1'].rootAssembly.Set(name='Set-5', referencePoints=(
    mdb.models['Model-1'].rootAssembly.referencePoints[RP4], ))#where load is applied
# Use it to constrain RBM
mdb.models['Model-1'].DisplacementBC(amplitude=UNSET, createStepName='Initial', 
    distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-2', 
    region=mdb.models['Model-1'].rootAssembly.sets['Set-5'], u1=SET, u2=SET, 
    u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=SET)# restrain UX, UY, allow UZ and all rota
# Module: Mesh (first, visualization commands)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
    
    
# Mesh Controls
mdb.models['Model-1'].rootAssembly.setMeshControls(elemShape=QUAD, regions=
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].faces.findAt(
    ((-b2/3, -(h - tf)/2, 2*L2/3), (0.0, -1.0, 0.0)), ((b2/3, -(h - tf)/2, L2/3), (0.0, -1.0, 0.0)), 
    ((-b2/3, (h - tf)/2, 2*L2/3), (0.0, -1.0, 0.0)), ((b2/3, (h - tf)/2, L2/3), (0.0, -1.0, 0.0)), 
    ((0.0, -(h - tf)/6, 2*L2/3), (-1.0, 0.0, 0.0)), ), technique=STRUCTURED)
    
mdb.models['Model-1'].rootAssembly.setElementType(elemTypes=(ElemType(
    elemCode=S8R, elemLibrary=STANDARD), ElemType(elemCode=STRI65, 
    elemLibrary=STANDARD)), regions=(
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'].faces.findAt(
    ((-b2/3, -(h - tf)/2, 2*L2/3), (0.0, -1.0, 0.0)), ((b2/3, -(h - tf)/2, L2/3), (0.0, -1.0, 0.0)), 
    ((-b2/3, (h - tf)/2, 2*L2/3), (0.0, -1.0, 0.0)), ((b2/3, (h - tf)/2, L2/3), (0.0, -1.0, 0.0)), 
    ((0.0, -(h - tf)/6, 2*L2/3), (-1.0, 0.0, 0.0)),), ))
    
mdb.models['Model-1'].rootAssembly.seedPartInstance(deviationFactor=0.1, 
    minSizeFactor=0.1, regions=(
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'], ), size=b/4)
mdb.models['Model-1'].rootAssembly.generateMesh(regions=(
    mdb.models['Model-1'].rootAssembly.instances['Part-1-1'], ))
# 
# >>> BUCKLE only. Not needed for RIKS. 
# Very picky because the BLOCk number is chosen manually on the .inp file
# Modify Keywords in input.file, using the GUI, so that it picks the correct BLOCK. 
# Add at the end of BUCKLE step, before **BOUNDARY CONDITIONS
# Copy the commands from the .rec file
# Note block: 38 writes mode shapes to .fil file
# Any change to the script above may change the block number. See textbook. 
mdb.models['Model-1'].keywordBlock.synchVersions(storeNodesAndElements=False)# !!!
mdb.models['Model-1'].keywordBlock.insert(41,'\n*NODE FILE, GLOBAL=YES, LAST MODE=10\nU')
# _____________________________________________________________________________________
# >>> RIKS only. Comment for BUCKLE. Uncomment for RIKS. 
# Modify Keywords in input.file, using the GUI, so that it picks the correct BLOCK. 
# Must add using Menu: Model, Edit-keywords, Model-1
# Must add before Step-2 (block 39 for this example but may change)
# Add more groups like "\n2,0.05\n3,0.05" etc. for more imperfections
#mdb.models['Model-1'].keywordBlock.synchVersions(storeNodesAndElements=False)# !!!
#mdb.models['Model-1'].keywordBlock.insert(39, 
#    '\n*IMPERFECTION, FILE=WF6X6, STEP=1\n1,0.05')#\n mode, magnitude (of imperfection)
# _____________________________________________________________________________________
# Module: Job (first, visualization commands)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(meshTechnique=OFF)
# 
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
    explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
    memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF, 
    multiprocessingMode=DEFAULT, name=jobname, nodalOutputPrecision=SINGLE, 
    numCpus=8, numDomains=8, numGPUs=0, queue=None, resultsFormat=ODB, scratch=
    '', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
# Submit
mdb.jobs[jobname].submit(consistencyChecking=OFF)
# Add this for the script to wait until the Job is done before doing post-processing
mdb.jobs[jobname].waitForCompletion()
#
#Post
# coordinates from Model Instance
# openMdb(pathName=jobname+'.cae') # already in memory
i=mdb.models['Model-1'].rootAssembly.instances['Part-1-1']
fd = open(jobname+"-nodes.npy", "w")#nunpy unformatted file
nodeArray = []#initialize node array
nNodes=len(i.nodes)# actual number
x_center = [];
y_center=[];
z_center=[];
el_center=[];
x_wave = [];
y_wave=[];
z_wave=[];
el_wave=[];
for node in range(nNodes):
    nodeLabel = i.nodes[node].label
    nodeCoord = list(i.nodes[node].coordinates)#list
    nodeArray.append([nodeCoord[0], nodeCoord[1], nodeCoord[2], ])#2D array
    if (abs(nodeCoord[0]) <=1e-12 and abs(nodeCoord[1]) <= 1e-12):
        el_center.append(nodeLabel)
        x_center.append(nodeCoord[0])
        y_center.append(nodeCoord[1])
        z_center.append(nodeCoord[2])
    elif (nodeCoord[0] <=-75.9 and nodeCoord[1] <= 72.8 and nodeCoord[2] >=0.):    
        el_wave.append(nodeLabel)
        x_wave.append(nodeCoord[0])
        y_wave.append(nodeCoord[1])
        z_wave.append(nodeCoord[2])
np.savetxt(fd,nodeArray)
fd.close()

centerline = np.column_stack([el_center,x_center,y_center,z_center])
wave = np.column_stack([el_wave,x_wave,y_wave,z_wave])

import re # regex support
import odbAccess
a=odbAccess.openOdb(jobname+'.odb',readOnly=True)
eigen = []#initialize list of eigenvalues
umax_list = []
umax1 = 0
umax2 = 0
umax3 = 0

for frm in range(nModes):# range(2): 0,1

    frame = frm + 1
    f = a.steps['Step-1'].frames[frame]# frame 1 -> mode 1
    # get the numerical value of the eigenvalue from fl.description
    description = re.findall(r'[\d\.]+',f.description)
    # mode = int(description[0])# same as frame
    try:
        eigen.append(float(description[1]+'E+'+description[2]))
    except:
        eigen.append(float(description[1]))#plain float value, no E+xx
    u_list1=[]
    u_list2=[]
    u_list3=[]
    strn = f.fieldOutputs['U'].values
    for node in range(len(strn)):
        u1 = f.fieldOutputs['U'].values[node].data[0]
        u_list1.append(abs(u1))
        u2 = f.fieldOutputs['U'].values[node].data[1]
        u_list2.append(abs(u2))
        u3 = f.fieldOutputs['U'].values[node].data[2]
        u_list3.append(abs(u3))
    try:
        umax_list.append([max(u_list1),max(u_list2),max(u_list3),float(description[1]+'E+'+description[2]),frame])      
    except:
        umax_list.append([max(u_list1),max(u_list2),max(u_list3),float(description[1]),frame])      
    if umax1 < max((u_list1)):
        umax1 = max((u_list1))
    if umax2 < max((u_list2)):
        umax2 = max((u_list2))  
    if umax3 < max((u_list3)):
        umax3 = max((u_list3))
    
umax = max(umax1,umax2,umax3)

for frm in range(nModes):# range(2): 0,1
    ucenter1 = []
    ucenter2 = []
    ucenter3 = []
    u_wave1=[]
    u_wave2=[]
    u_wave3=[]
    frame = frm + 1
    f = a.steps['Step-1'].frames[frame]# frame 1 -> mode 1
    for node in wave[:,0]:
        u1 = f.fieldOutputs['U'].values[int(node)].data[0]
        u_wave1.append((u1))
        u2 = f.fieldOutputs['U'].values[int(node)].data[1]
        u_wave2.append((u2))
        u3 = f.fieldOutputs['U'].values[int(node)].data[2]
        u_wave3.append((u3))
    for node in centerline[:,0]:
        u1 = f.fieldOutputs['U'].values[int(node)].data[0]
        ucenter1.append((u1))
        u2 = f.fieldOutputs['U'].values[int(node)].data[1]
        ucenter2.append((u2))
        u3 = f.fieldOutputs['U'].values[int(node)].data[2]
        ucenter3.append((u3))
        
    fr_wave = np.column_stack([el_wave,x_wave,y_wave,z_wave,u_wave1,u_wave2,u_wave3])
    fr_center = np.column_stack([el_center,x_center,y_center,z_center,ucenter1,ucenter2,ucenter3])
    np.savetxt(str(lambd)+'_wave_by_frame_'+str(frame)+'.txt',fr_wave)
    np.savetxt(str(lambd)+'_center_by_frame_'+str(frame)+'.txt',fr_center)


np.savetxt('deformations_'+str(num)+'.txt',umax_list)
s = ' '.join(['%g']*len(eigen))+'\n'
fo.write ('L = %g\n' % (L))
fo.write (s % tuple(eigen))
fo.close()
session.odbs[jobname+'.odb'].close()
